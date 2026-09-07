import json
import multiprocessing
import os
import time
from collections import Counter
from collections.abc import Iterator
from typing import BinaryIO

import regex as re

type Pair = tuple[bytes, bytes]
type Vocab = dict[int, bytes]
type Merges = list[tuple[bytes, bytes]]

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
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

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def per_core_pre_tokenization(file_path, start: int, end: int, special_tokens: list[str]) -> Counter:
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        return Counter(_pre_tokenize(chunk, special_tokens))


def _pre_tokenize(chunk: str, special_tokens: list[str]) -> Iterator[str]:
    for part in re.split("|".join(special_tokens), chunk):
        for m in re.finditer(PAT, part):
            if m:
                full_hit = m.group(0)
                yield full_hit


def _identify_merge(pair_counts: Counter[Pair]) -> Pair | None:
    max_freq = pair_counts.most_common(1)[0][1]
    if max_freq <= 0:
        return None

    best_candidates = [pair for pair, freq in pair_counts.items() if freq == max_freq]
    return max(best_candidates)


def bpe_naive(token_freqs: Counter, vocab_size: int, special_tokens: list[str]) -> tuple[Vocab, Merges]:
    """
    Performs naive Byte-Level BPE on pre-tokenized frequencies.
    Returns the vocabulary mapping of token_id -> bytes.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = list()

    bytes_tuple_table: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in token_freqs.items()
    }

    # actual vocab size is smaller due to reserved special tokens
    while len(vocab) < vocab_size - len(special_tokens):
        pair_counts: Counter[Pair] = Counter()

        for word_tuple, count in bytes_tuple_table.items():
            for pair in zip(word_tuple, word_tuple[1:]):
                pair_counts[pair] += count

        if not pair_counts:
            break

        best_pair = _identify_merge(pair_counts)
        if not best_pair:
            break

        merges.append(best_pair)
        new_vocb = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_vocb

        new_table = {}
        for word_tuple, count in bytes_tuple_table.items():
            new_tuple = list()
            i = 0
            while i < len(word_tuple):
                if i < len(word_tuple) - 1 and (word_tuple[i], word_tuple[i + 1]) == best_pair:
                    new_tuple.append(new_vocb)
                    i += 2
                else:
                    new_tuple.append(word_tuple[i])
                    i += 1
            new_table[tuple(new_tuple)] = count

        bytes_tuple_table = new_table

    # adding back special tokens to vocab
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token.encode("utf-8"))

    return vocab, merges


if __name__ == "__main__":
    num_cores = multiprocessing.cpu_count()
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10_000

    start_ts = time.time()
    print(start_ts)

    ## Usage
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_cores, "|".join(special_tokens).encode("utf-8"))
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        token_counts = Counter()

        print(f"Produced {len(boundaries)} boundaries, to be processed with {num_cores} cores")
        with multiprocessing.Pool(processes=num_cores) as pool:
            # 2. Distribute the work using map
            # This blocks until the entire list is processed
            results = pool.starmap(
                per_core_pre_tokenization,
                [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )

            for c in results:
                token_counts += c

        vocab, merges = bpe_naive(token_counts, vocab_size, special_tokens)

        end_ts = int(time.time())
        print(f"Time taken: {(end_ts - start_ts) // 1000}")

    with open("./output/vocab.json", "w") as f:
        json.dump(vocab, f)
    with open("./output/merges.json", "w") as f:
        json.dump(merges, f)
