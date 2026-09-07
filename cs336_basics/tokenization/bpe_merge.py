from collections import Counter, defaultdict
from collections.abc import Iterator

import regex as re

from cs336_basics.tokenization.datamodel import Merges, Pair, Vocab

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


def _build_merged_word(affected_word_tuple: tuple[bytes, ...], pair_to_merge: Pair) -> tuple[bytes, ...]:
    new_tuple = list()
    i = 0
    new_vocb = pair_to_merge[0] + pair_to_merge[1]
    while i < len(affected_word_tuple):
        if i < len(affected_word_tuple) - 1 and (affected_word_tuple[i], affected_word_tuple[i + 1]) == pair_to_merge:
            new_tuple.append(new_vocb)
            i += 2
        else:
            new_tuple.append(affected_word_tuple[i])
            i += 1

    return tuple(new_tuple)


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
            new_tuple = _build_merged_word(word_tuple, best_pair)
            new_table[tuple(new_tuple)] = count

        bytes_tuple_table = new_table

    # adding back special tokens to vocab
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token.encode("utf-8"))

    return vocab, merges


def bpe_cached(token_freqs: Counter, vocab_size: int, special_tokens: list[str]) -> tuple[Vocab, Merges]:
    """
    Performs naive Byte-Level BPE on pre-tokenized frequencies.
    Returns the vocabulary mapping of token_id -> bytes.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    merges: list[Pair] = list()

    word_table: dict[tuple[bytes, ...], int] = {
        tuple(bytes([b]) for b in token.encode("utf-8")): count for token, count in token_freqs.items()
    }

    pair_counts: Counter[Pair] = Counter()
    pair_to_words_cache: dict[Pair, set[tuple[bytes, ...]]] = defaultdict(set)

    for word_tuple, count in word_table.items():
        for pair in zip(word_tuple, word_tuple[1:]):
            pair_counts[pair] += count

            # cache mapping to word_tuple
            pair_to_words_cache[pair].add(word_tuple)

    # actual vocab size is smaller due to reserved special tokens
    while len(vocab) < vocab_size - len(special_tokens):
        best_pair = _identify_merge(pair_counts)
        if not best_pair:
            break

        merges.append(best_pair)
        new_vocb = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = new_vocb

        for affected_word in list(pair_to_words_cache.get(best_pair)):
            word_count = word_table[affected_word]
            new_word = _build_merged_word(affected_word, best_pair)

            # update raw word table
            word_table[new_word] = word_count
            del word_table[affected_word]

            # update pair table
            for old_pair in zip(affected_word, affected_word[1:]):
                pair_counts[old_pair] -= word_count

                if affected_word in pair_to_words_cache[old_pair]:
                    pair_to_words_cache[old_pair].remove(affected_word)

            for new_pair in zip(new_word, new_word[1:]):
                pair_counts[new_pair] += word_count
                pair_to_words_cache[new_pair].add(new_word)

    # adding back special tokens to vocab
    for special_token in special_tokens:
        vocab[len(vocab)] = bytes(special_token.encode("utf-8"))

    return vocab, merges
