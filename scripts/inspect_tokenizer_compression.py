import json
import logging
import mmap
import statistics
from collections.abc import Iterator

import regex as re

from cs336_basics.tokenization.tokenizer import Tokenizer

TOKENIZERS = [
    (
        "output/TinyStoriesV2-GPT4-train-vocab_20260913102314.json",
        "output/TinyStoriesV2-GPT4-train-merge_20260913102314.json",
    ),
    (
        "output/owt_train-vocab_20260913115756.json",
        "output/owt_train-merge_20260913115756.json",
    ),
]

SAMPLE_DOCS_SOURCE_FILE = ("data/owt_train.txt", "data/TinyStoriesV2-GPT4-train.txt")

SPECIAL_TOKENS = ["<|endoftext|>"]


def _sample_documents(file_path: str, special_tokens: list[str], max_docs: int | None = None) -> Iterator[str]:
    special_tokens = sorted(special_tokens, key=lambda token: -len(token))
    pattern = "|".join(re.escape(d) for d in special_tokens)
    pattern_bytes = pattern.encode("utf-8")
    compiled_pattern = re.compile(pattern_bytes)

    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            positions = [match.start() for match in compiled_pattern.finditer(mm)]

            prev_idx = 0
            docs = 0
            for index in positions:
                yield mm[prev_idx:index].decode("utf-8")
                docs += 1
                prev_idx = index

                if docs % 10000 == 0:
                    logging.info(f"Processed {docs} docs in {file_path}")

                if max_docs is not None and docs >= max_docs:
                    break


def main(vocab_path: str, merge_path: str, special_tokens: list[str] = SPECIAL_TOKENS) -> None:
    logging.info(f"Loading tokenizer from {vocab_path} and {merge_path}")
    tokenizer = Tokenizer.from_files(
        vocab_path,
        merge_path,
    )

    res = dict()
    res_summary = dict()
    for sample_file in SAMPLE_DOCS_SOURCE_FILE:
        logging.info(f"Loading sample doc from {sample_file}")

        ratios = list()
        for idx, doc in enumerate(_sample_documents(sample_file, special_tokens=special_tokens, max_docs=10)):
            if not doc:
                continue

            doc_in_bytes = doc.encode("utf-8")
            doc_in_tokens = tokenizer.encode(doc)

            ratios.append(len(doc_in_bytes) / len(doc_in_tokens))

        res[sample_file] = ratios
        res_summary[sample_file] = {"mean": statistics.mean(ratios), "std": statistics.stdev(ratios)}

    return res_summary, res


if __name__ == "__main__":
    for vocab_path, merge_path in TOKENIZERS:
        summary, raw = main(vocab_path, merge_path)
        print(json.dumps(summary, indent=4))
