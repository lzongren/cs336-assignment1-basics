import logging
import multiprocessing
import os
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap

from cs336_basics.tokenization.pre_tokenization import find_chunk_boundaries
from cs336_basics.tokenization.tokenizer import Tokenizer

type TokenizerFromFiles = tuple[str, str]

TOKENIZERS_N_DATA = {
    (
        "output/TinyStoriesV2-GPT4-train-vocab_20260913102314.json",
        "output/TinyStoriesV2-GPT4-train-merge_20260913102314.json",
    ): (
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/TinyStoriesV2-GPT4-valid.txt",
    ),
    (
        "output/owt_train-vocab_20260913115756.json",
        "output/owt_train-merge_20260913115756.json",
    ): (
        "data/owt_train.txt",
        "data/owt_valid.txt",
    ),
}

SPECIAL_TOKENS = ["<|endoftext|>"]


def per_core_encoding(
    chunk_index: int,
    tokenizer_from_files: TokenizerFromFiles,
    special_tokens: list[str],
    file_path: str,
    start: int,
    end: int,
    output_dir: str | os.PathLike,
) -> tuple[int, str, int]:
    """
    Each worker returns (chunk index, filename, token_count)
    """
    tokenizer = Tokenizer.from_files(*tokenizer_from_files, special_tokens)

    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

        encoding_iterable = tokenizer.encode(chunk)
        data = np.fromiter(encoding_iterable, dtype="uint16")

    output_path = Path(output_dir) / f"chunk_{chunk_index}.npy"
    np.save(output_path, data)

    res = chunk_index, str(output_path), len(data)
    logging.info(f"Chunk {chunk_index} worker completed: {res}")
    return res


def merge_shards_to_memmap(results: list[tuple[int, str, int]], final_output_path: Path) -> None:
    """
    Combines worker chunk files into a single master .npy file using memory mapping.
    Ensures memory consumption remains small regardless of dataset size.
    """
    sorted_results = sorted(results, key=lambda x: x[0])

    total_tokens = sum(token_count for _, _, token_count in sorted_results)
    logging.info(f"Merging {len(sorted_results)} shards. Total tokens: {total_tokens:,}")

    final_mmap = open_memmap(filename=final_output_path, mode="w+", dtype="uint16", shape=(total_tokens,))

    current_offset = 0
    for chunk_idx, filepath, token_count in sorted_results:
        if token_count == 0:
            continue

        logging.info(f"Slicing chunk_{chunk_idx} into index [{current_offset}:{current_offset + token_count}]")

        # Open individual worker shard in read-only memmap mode
        src_mmap = np.load(filepath, mmap_mode="r")

        # Write directly to the destination memory slice
        final_mmap[current_offset : current_offset + token_count] = src_mmap

        current_offset += token_count

    final_mmap.flush()
    del final_mmap
    logging.info(f"Successfully created final memory map at {final_output_path}")


def main(
    vocab_path: str,
    merge_path: str,
    data_files: list[str],
    num_cores: int,
    num_tasks: int,
    special_tokens: list[str] = SPECIAL_TOKENS,
) -> None:
    logging.info(f"Loading tokenizer from {vocab_path} and {merge_path}")

    for data_file in data_files:
        logging.info(f"Loading data file {data_file}")

        with open(data_file, "rb") as f:
            boundaries = find_chunk_boundaries(f, num_tasks, "|".join(special_tokens).encode("utf-8"))
            logging.info(f"Produced {len(boundaries)} boundaries, to be processed with {num_cores} cores")

            output_dir = Path(f"output/{os.path.basename(data_file).split('.', maxsplit=1)[0]}")
            output_dir.mkdir(parents=True, exist_ok=True)

            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.starmap(
                    per_core_encoding,
                    [
                        (
                            idx,
                            (vocab_path, merge_path),
                            special_tokens,
                            data_file,
                            start,
                            end,
                            output_dir,
                        )
                        for idx, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:]))
                    ],
                )

            final_npy_name = f"{os.path.basename(data_file).split('.', maxsplit=1)[0]}_final.npy"
            final_output_path = output_dir / final_npy_name

            merge_shards_to_memmap(results, final_output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for (vocab_path, merge_path), datafiles in TOKENIZERS_N_DATA.items():
        vocab_basename = os.path.basename(vocab_path)
        tokenizer_name = vocab_basename.split(".", maxsplit=1)[0].split("-", maxsplit=1)[0]

        res = main(vocab_path, merge_path, datafiles, 16, 160, SPECIAL_TOKENS)
