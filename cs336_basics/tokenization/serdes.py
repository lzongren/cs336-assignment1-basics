import json
from datetime import datetime
from enum import Enum
from os import PathLike

from cs336_basics.tokenization.datamodel import Merges, Vocab


class MergeOrVocab(Enum):
    MERGE = "merge"
    VOCAB = "vocab"


class BytesEncoder(json.JSONEncoder):
    def default(self, obj: object):
        if isinstance(obj, bytes):
            return list(obj)
        return super().default(obj)


def serdes_filename(dataset_label: str, merge_or_vocab: MergeOrVocab) -> str:
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    return f"{dataset_label}-{merge_or_vocab.value}_{current_timestamp}"


def vocab_from_file(file_path: str | PathLike) -> Vocab:
    with open(file_path) as f:
        data = json.load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Vocab should be a dict, invalid f{file_path}")

        return {i: bytes(l) for i, l in data.items()}


def merges_from_file(file_path: str | PathLike) -> Merges:
    with open(file_path) as f:
        data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Vocab should be a list of tuples, invalid f{file_path}")

        return [(bytes(item[0]), bytes(item[1])) for item in data]


if __name__ == "__main__":
    vocab = vocab_from_file("output/TinyStoriesV2-GPT4-train-vocab_20260913102314.json")
    merges = merges_from_file("output/TinyStoriesV2-GPT4-train-merge_20260913102314.json")
