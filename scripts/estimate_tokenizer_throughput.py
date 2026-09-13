import logging
import math
import time

from cs336_basics.tokenization.tokenizer import Tokenizer
from scripts.inspect_tokenizer_compression import _sample_documents

TOKENIZER = (
    "output/owt_train-vocab_20260913115756.json",
    "output/owt_train-merge_20260913115756.json",
)

SAMPLE_DOCS_SOURCE_FILE = ("data/owt_train.txt", "data/TinyStoriesV2-GPT4-train.txt")

SPECIAL_TOKENS = ["<|endoftext|>"]


def main(vocab_path: str, merge_path: str, special_tokens: list[str] = SPECIAL_TOKENS) -> None:
    logging.info(f"Loading tokenizer from {vocab_path} and {merge_path}")
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merge_path,
    )

    total_bytes = 0
    total_time = 0

    for sample_file in SAMPLE_DOCS_SOURCE_FILE:
        logging.info(f"Loading sample doc from {sample_file}")

        for idx, doc in enumerate(_sample_documents(sample_file, special_tokens=special_tokens, max_docs=1000)):
            if not doc:
                continue

            start_ts = time.time()
            tokenizer.encode(doc)
            total_time += time.time() - start_ts

            doc_in_bytes = doc.encode("utf-8")
            total_bytes += len(doc_in_bytes)

    return total_bytes, total_time


if __name__ == "__main__":
    total_bytes, duration_sec = main(*TOKENIZER)
    throughput = total_bytes / duration_sec
    print(f"{total_bytes / duration_sec:.2f} bytes/second")

    pi_dateset_size = 825 * math.pow(10, 9)

    duration = pi_dateset_size // throughput
    print(f"Time take to encode Pi dataset: {duration}")
