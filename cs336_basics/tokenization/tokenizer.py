from collections.abc import Iterable, Iterator

from cs336_basics.tokenization.bpe_merge import _pre_tokenize
from cs336_basics.tokenization.datamodel import Merges, Vocab


class Tokenizer:
    def __init__(self, vocab: Vocab, merges: Merges, special_tokens: list[str] | None = None) -> None:
        self.vocab = vocab
        self.vocab_reversed = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}
        self.merges = merges
        self.merges_dict = {pair: i for i, pair in enumerate(self.merges)}
        self.special_tokens = special_tokens or list()

        # ensure reverse look-up is not losing data
        assert len(self.merges) == len(self.merges_dict)

        # ensure special tokens are inside vocab
        for special_token in self.special_tokens:
            if bytes(special_token, encoding="utf-8") not in self.vocab_reversed:
                raise ValueError(f"Vocab does not contain special token {special_token}")

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> None:
        pass

    def encode(self, text: str) -> list[int]:
        return list(self._encode_iterable(text))

    def _encode_iterable(self, text: str) -> Iterable[int]:
        for word_str in _pre_tokenize(text, self.special_tokens, preserve_delimiter=True):
            if word_str in self.special_tokens:
                word_bytes = bytes(word_str, encoding="utf-8")
                yield self.vocab_reversed[word_bytes]

            else:
                word_ints = list(bytes(word_str, encoding="utf-8"))
                # converting integer to bytes should use bytes([...]), otherwise bytes(...) just means
                # "creating this many zero bytes"
                word_bytes = tuple([bytes([part]) for part in word_ints])

                processed_word = self._build_merged_word(word_bytes)
                for part_bytes in processed_word:
                    yield self.vocab_reversed[part_bytes]

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self._encode_iterable(chunk)

    def decode(self, encoding: list[int]) -> str:
        res = b""

        for item_id in encoding:
            item_bytes = self.vocab[item_id]
            res += item_bytes

        return res.decode("utf-8", errors="replace")

    def _build_merged_word(self, word_tuple: tuple[bytes, ...]) -> tuple[bytes, ...]:
        assert isinstance(word_tuple, tuple)

        while True:
            merged = False
            new_list = list()

            possible_merges_by_rank = list()
            for i in range(len(word_tuple) - 1):
                if (pair := (word_tuple[i], word_tuple[i + 1])) in self.merges_dict:
                    # tuple: (rank, index of the merge)
                    possible_merges_by_rank.append((self.merges_dict[pair], i))

            if not possible_merges_by_rank:
                break

            _, index_to_merge = min(possible_merges_by_rank)

            i = 0
            while i < len(word_tuple):
                if i == index_to_merge:
                    new_list.append(word_tuple[i] + word_tuple[i + 1])
                    merged = True
                    i += 2
                else:
                    new_list.append(word_tuple[i])
                    i += 1

            word_tuple = tuple(new_list)
            if not merged:
                break

        return word_tuple
