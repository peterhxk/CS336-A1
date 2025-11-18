from multiprocessing import Pool, Process
from collections import defaultdict
import os
from typing import BinaryIO

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
    binary_special_tokens = [token.encode("utf-8") for token in special_tokens]

    with open(input_path, "rb") as f:
        num_processes = 5
        boundaries = find_chunk_boundaries(f, num_processes, binary_special_tokens)
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        args.append((input_path, start, end, binary_special_tokens))
    with Pool(len(args)) as pool:
        results = pool.starmap(count_pair, args)
    total_count = defaultdict(int)
    for count_map in results:
        for k, v in count_map.items():
            total_count[k] += v
    print(total_count)
    return

def count_pair(input_path, start, end, binary_special_tokens):
    """
        Counts byte pairs
    """
    with open(input_path, "rb") as f:
        f.seek(start)
        init_chunk = f.read(end-start)
    splited_chunks = [init_chunk]
    for t in binary_special_tokens:
        for i in range(len(splited_chunks)):
            arr = splited_chunks[i].split(t)
            splited_chunks[i] = arr[0]
            for chunk in arr[1:]:
                splited_chunks.append(chunk)
    count_map = defaultdict(int)
    for splited_chunk in splited_chunks:
        for i, j in zip(splited_chunk, splited_chunk[1:]):
            count_map[(i, j)] += 1

    return count_map




if __name__ == "__main__":
    train_bpe("../../data/TinyStoriesV2-GPT4-valid.txt",5,["<|endoftext|>"])