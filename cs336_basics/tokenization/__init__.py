import multiprocessing
import os
from collections import Counter

from cs336_basics.tokenization.bpe_merge import bpe_cached, bpe_naive, per_core_pre_tokenization
from cs336_basics.tokenization.pre_tokenization import find_chunk_boundaries


def train_bpe(
    input_path: str | os.PathLike,  # noqa: F821
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    num_cores = multiprocessing.cpu_count()

    ## Usage
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_cores, "|".join(special_tokens).encode("utf-8"))

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        token_counts = Counter()

        with multiprocessing.Pool(processes=num_cores) as pool:
            # 2. Distribute the work using map
            # This blocks until the entire list is processed
            results = pool.starmap(
                per_core_pre_tokenization,
                [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )

            for c in results:
                token_counts += c

        return bpe_cached(token_counts, vocab_size, special_tokens)
