import json
import logging
import multiprocessing
import os
import time
from collections import Counter
from os import PathLike
from pathlib import Path

from cs336_basics.tokenization import serdes
from cs336_basics.tokenization.bpe_merge import bpe_cached, per_core_pre_tokenization
from cs336_basics.tokenization.datamodel import Merges, Vocab
from cs336_basics.tokenization.pre_tokenization import find_chunk_boundaries


class BPETrainer:
    def __init__(
        self,
        input_path: str | PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        output_folder: str | PathLike | None = None,
        num_cores: int | None = None,
        data_label: str | None = None,
    ):
        if not input_path or not Path(input_path).is_file():
            raise ValueError(f"Invalid file path to perform BPE training {input_path}")

        self.input_path = input_path
        self.output_folder = output_folder
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or list()
        self.num_cores = int(multiprocessing.cpu_count() * 0.8) if not num_cores else num_cores

        # generate data label from dataset
        self.data_label = data_label
        if not self.data_label:
            training_file = os.path.basename(self.input_path)
            self.data_label = training_file.split(".", maxsplit=1)[0]

    # @memory_stats
    def __call__(self, *args, **kwds) -> tuple[Vocab, Merges]:
        start_ts = time.time()

        with open(self.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.num_cores * 10, "|".join(self.special_tokens).encode("utf-8"))

            chunking_ts = time.time()
            logging.info(f"Finding boundaries complete: {(chunking_ts - start_ts):.2f} seconds")
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            token_counts = Counter()

            logging.info(f"Produced {len(boundaries)} boundaries, to be processed with {self.num_cores} cores")
            with multiprocessing.Pool(processes=self.num_cores) as pool:
                # 2. Distribute the work using map
                # This blocks until the entire list is processed
                results = pool.starmap(
                    per_core_pre_tokenization,
                    [
                        (self.input_path, start, end, self.special_tokens)
                        for start, end in zip(boundaries[:-1], boundaries[1:])
                    ],
                )

                for c in results:
                    token_counts += c

                parallel_tokenization_ts = time.time()
                logging.info(
                    f"Pre-tokenization completes (to start BPE): {(parallel_tokenization_ts - chunking_ts):.2f} seconds"
                )

            vocab, merges = bpe_cached(token_counts, self.vocab_size, self.special_tokens)

            end_ts = int(time.time())
            logging.info(f"BPE completes: {(end_ts - parallel_tokenization_ts):.2f} seconds")
            logging.info(f"Completes in total: {(end_ts - start_ts):.2f} seconds")

            longest_token = None
            for token in vocab.values():
                if not longest_token or len(token) > len(longest_token):
                    longest_token = token

            logging.info(f"Longest token is {len(longest_token)}: {list(longest_token)}")

            if self.output_folder:
                self._serialize(vocab, merges)

            return vocab, merges

    def _serialize(self, vocab: Vocab, merges: Merges) -> None:
        if not self.output_folder:
            raise ValueError("Unspecified persistence path!")

        vocab_basename = serdes.serdes_filename(self.data_label, serdes.MergeOrVocab.VOCAB)
        merge_basename = serdes.serdes_filename(self.data_label, serdes.MergeOrVocab.MERGE)

        vocab_path = Path(self.output_folder) / f"{vocab_basename}.json"
        merge_path = Path(self.output_folder) / f"{merge_basename}.json"

        logging.info(f"Serializing vocab to {vocab_path} and merge to {merge_path}")

        with open(vocab_path, "w") as f:
            json.dump(vocab, f, cls=serdes.BytesEncoder)
        with open(merge_path, "w") as f:
            json.dump(merges, f, cls=serdes.BytesEncoder)


if __name__ == "__main__":
    # trainer = BPETrainer("data/TinyStoriesV2-GPT4-train.txt", 10_000, ["<|endoftext|>"], "output/")
    trainer = BPETrainer("data/owt_train.txt", 32_000, ["<|endoftext|>"], "output/")
    trainer()
