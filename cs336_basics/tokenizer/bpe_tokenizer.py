from multiprocessing import Pool, Process
from collections import defaultdict
import os
from typing import BinaryIO
import time

def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    for split_special_token in split_special_tokens:
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find one of the special token in the mini chunk
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    break
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def parallel_count(indices, boundaries, binary_special_tokens):
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append((indices[start:end], binary_special_tokens))

    with Pool(min(len(args), os.cpu_count())) as pool:
        results = pool.starmap(count_pair, args)
    
    total_count = defaultdict(int)
    for count_map in results:
        for k, v in count_map.items():
            total_count[k] += v
    return total_count
    
def count_pair(chunk:list[int] , binary_special_tokens:list[bytes]):
    """
        Counts byte pairs used for parallel_count
    """
    count_map = defaultdict(int)
    for i, j in zip(chunk, chunk[1:]):
        count_map[(i, j)] += 1

    return count_map

def parallel_merge(indices, boundaries, pair, new_index) -> tuple[list[int],list[int]]:
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append((indices[start:end], pair, new_index))
    with Pool(min(len(args), os.cpu_count())) as pool:
        results = pool.starmap(compute_merge, args)
    
    new_body = []
    for i, (chunk, offset) in enumerate(results):
        new_body += chunk
        boundaries[i+1] -= offset
    return new_body, boundaries

def compute_merge(chunk, pair, new_ind) -> tuple[list[int], int]:
    """
    Docstring for compute_merge
    Merges chunks with new index used for parallel_merge
    
    :param chunk: Description
    :param pair: Description
    :param new_ind: Description
    :return: Description
    :rtype: tuple[list[int], int]
    """
    i, j = 0, 0
    ind1, ind2 = pair
    while i < len(chunk):
        if i+1 == len(chunk):
            chunk[j] = chunk[i]
            j += 1
            i += 1
        elif chunk[i] == ind1 and chunk[i+1] == ind2:
            chunk[j] = new_ind
            i += 2
            j += 1
        else:
            chunk[j] = chunk[i]
            i += 1
            j += 1
    new_chunk = chunk[:j]
    offset = i-j
    return new_chunk, offset



def train_bpe( input_path:str,
               vocab_size:int,
               special_tokens: list[str]
               ) -> tuple[dict[int, bytes], list[tuple[bytes,bytes]]] :
    """
    input:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary,
        vocabulary items produced from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special tokens do not otherwize affect BPE training.
    return:
        vocab: the tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes)
        merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>),
        representing that <token1> was merged with <token2>. The merges should be ordered by order of creation
    """
    merges: dict[tuple[int,int],int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}
    binary_special_tokens = [token.encode("utf-8") for token in special_tokens]
    remaining_vocab_size = vocab_size-256-len(special_tokens)

    stamp("Starting tokenizer")

    with open(input_path, "rb") as f:
        num_processes = 5
        boundaries = find_chunk_boundaries(f, num_processes, binary_special_tokens)
        f.seek(0)
        body = f.read()
        indices = list(body)
    
    stamp(f"Running BPE tokenizer with {min(len(boundaries)-1, os.cpu_count())} processors")


    for i in range(remaining_vocab_size):
        iter_start = time.time()

        total_count = parallel_count(indices, boundaries, binary_special_tokens)
        most_common_pair = max(total_count, key=total_count.get)
        ind1,ind2 = most_common_pair
        
        stamp(f"[{i+1}/{remaining_vocab_size}] Most common pair: {most_common_pair}")


        #merge that pair
        new_index = 256 + i
        merges[most_common_pair] = new_index
        vocab[new_index] = vocab[ind1]+vocab[ind2]
        indices, boundaries = parallel_merge(indices, boundaries, most_common_pair, new_index)
        
        iter_end = time.time()
        stamp(f"Iteration {i+1} completed in {iter_end - iter_start:.2f}s")


    stamp(f"result: {vocab}, {merges}")
    return vocab, merges



if __name__ == "__main__":
    train_bpe("../../data/TinyStoriesV2-GPT4-valid.txt",300,["<|endoftext|>"])