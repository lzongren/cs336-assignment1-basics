
## Problem (unicode1): Understanding Unicode (1 point)

#### (a) What Unicode character does `chr(0)` return?

It returns `'\x00'`.

#### (b) How does this character’s string representation (__repr__()) differ from its printed representation?

It returns its string form as `"'\\x00'"` and `\` is escaped.

#### (c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations.

It is not printable, in the printed form not distinguishable from empty string.

```
>>> "this is a test" + chr(0) + "string"
'this is a test\x00string'
>>> print("this is a test" + chr(0) + "string")
this is a teststring
```

## Problem (unicode2): Unicode Encodings (3 points)

#### (a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings?

For LM we're training on great sources of inputs incl. non-ascii inputs, although UTF-8, UTF-16 and UTF-32 are different ways of
representing unicode code points and can support all the different inputs, UTF-8 is more concise in terms of numerical representation, which may presents as an advantage in terms of amount of matrix or vector computations later on.

```
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> utf16_encoded = test_string.encode("utf-16")
>>> utf32_encoded = test_string.encode("utf-32")
>>> print(len(utf8_encoded))
23
>>> print(len(utf16_encoded))
28
>>> print(len(utf32_encoded))
56
```

#### (b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte sting that yields incorrect results.

```
>>> def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
...   return "".join([bytes([b]).decode("utf-8") for b in bytestring])
...   
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

This implementation is wrong because it in facts only handled the UTF-8's handling of western languages as the ascii characters only require the first byte in UTF-8, that's why decoding of encoded version of `hello` works. In reality, many non-ascii characters require to leverage UTF-8's encoding mechanism to look forward next bytes, of which the current implementation breaks.

```
>>> decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8"))
Traceback (most recent call last):
  File "<python-input-17>", line 1, in <module>
    decode_utf8_bytes_to_str_wrong("hello! こんにちは!".encode("utf-8"))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "<python-input-10>", line 3, in decode_utf8_bytes_to_str_wrong
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
                    ~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe3 in position 0: unexpected end of data
```
#### (c) Give a two-byte sequence that does not decode to any Unicode character(s).

```
>>> b'\xff\xff'.decode("utf-8")
Traceback (most recent call last):
  File "<python-input-29>", line 1, in <module>
    b'\xff\xff'.decode("utf-8")
    ~~~~~~~~~~~~~~~~~~^^^^^^^^^
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte
```

## Problem (train_bpe): BPE Tokenizer Training (15 points)
**Deliverable**: Write a function that, given a path to an input text file, trains a (byte-level) BPE
tokenizer. Your BPE training function should handle (at least) the following input parameters:

#### Input
* input_path: `str` Path to a text file with BPE tokenizer training data.
* vocab_size: `int` A positive integer that defines the maximum final vocabulary size (including
the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
* special_tokens: `list[str]` A list of strings to add to the vocabulary. During training, treat 
them as hard boundaries that prevent merges across their spans, but do not include them when
computing merge statistics.

Your BPE training function should return the resulting vocabulary and merges:

#### Output
* vocab: `dict[int, bytes]` The tokenizer vocabulary, a mapping from int (token ID in the
vocabulary) to bytes (token bytes).
* merges: `list[tuple[bytes, bytes]]` A list of BPE merges produced from training. Each list
item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with
<token2>. The merges should be ordered by order of creation.

To test your BPE training function against our provided tests, you will first need to implement
the test adapter at [adapters.run_train_bpe] . Then, run `uv run pytest tests/test_train_bpe.py`. Your implementation should be able to pass all tests. Optionally (this could
be a large time-investment), you can implement the key parts of your training method using some
systems language, for instance C++ (consider cppyy or nanobind) or Rust (using PyO3). If you do
this, be aware of which operations require copying vs reading directly from Python memory, and
make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note
that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most
that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but
the regex package in Python is, if anything, even faster





---
## Appendix

### Pytorch Environment
```
  - pyenv 2.8.4 configured in /home/zongr/.zshrc
  - Project Python: Miniforge Python 3.13.13
  - Project selection: assignment1-basics/.python-version
  - Environment: /home/zongr/Workspace/assignment1-basics/.venv
  - PyTorch 2.11.0+cu130
  - CUDA runtime 13.0
  - RTX 5080 detected with compute capability 12.0 and native sm_120
  - CUDA 1024×1024 matrix multiplication passed
  - pip check: no broken dependencies

  Use it with:

  cd ~/Workspace/assignment1-basics
  source .venv/bin/activate
  python
```