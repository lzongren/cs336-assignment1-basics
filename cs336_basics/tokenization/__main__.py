import json
import multiprocessing
import time
import tracemalloc
from collections import Counter

from cs336_basics.tokenization.bpe_merge import bpe_cached, per_core_pre_tokenization
from cs336_basics.tokenization.pre_tokenization import find_chunk_boundaries


class BytesEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, bytes):
            return list(obj)
        return super().default(obj)


# num_cores = multiprocessing.cpu_count()
# input_path = "data/TinyStoriesV2-GPT4-train.txt"
# vocab_size = 10_000
num_cores = int(multiprocessing.cpu_count() * 0.8)
input_path = "data/owt_train.txt"
vocab_size = 32_000
special_tokens = ["<|endoftext|>"]

start_ts = time.time()
tracemalloc.start()


## Usage
with open(input_path, "rb") as f:
    boundaries = find_chunk_boundaries(f, num_cores * 10, "|".join(special_tokens).encode("utf-8"))

    chunking_ts = time.time()
    print(f"Finding boundaries complete: {(chunking_ts - start_ts):.2f} seconds")
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

        parallel_tokenization_ts = time.time()
        print(f"Pre-tokenization completes (to start BPE): {(parallel_tokenization_ts - chunking_ts):.2f} seconds")

    vocab, merges = bpe_cached(token_counts, vocab_size, special_tokens)

    end_ts = int(time.time())
    print(f"BPE completes: {(end_ts - parallel_tokenization_ts):.2f} seconds")
    print(f"Completes in total: {(end_ts - start_ts):.2f} seconds")

    longest_token = None
    for token in vocab.values():
        if not longest_token or len(token) > len(longest_token):
            longest_token = token

    print(f"Longest token is {len(longest_token)}: {list(longest_token)}")

with open("./output/vocab.json", "w") as f:
    json.dump(vocab, f, cls=BytesEncoder)
with open("./output/merges.json", "w") as f:
    json.dump(merges, f, cls=BytesEncoder)

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 10**6:.2f} MB")
print(f"Peak memory usage:    {peak / 10**6:.2f} MB")
