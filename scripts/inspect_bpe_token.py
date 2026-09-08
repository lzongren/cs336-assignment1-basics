import mmap
import re

TINY_STORIES_LONGEST_TOKEN = [32, 97, 99, 99, 111, 109, 112, 108, 105, 115, 104, 109, 101, 110, 116]

OWT_LONGEST_LONGEST_TOKEN = [
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
    195,
    131,
    195,
    130,
]

if __name__ == "__main__":
    WINDOW_SIZE = 200
    for input_path, longest_token in [
        ("data/TinyStoriesV2-GPT4-train.txt", TINY_STORIES_LONGEST_TOKEN),
        ("data/owt_train.txt", OWT_LONGEST_LONGEST_TOKEN),
    ]:
        search_sequence = bytes(longest_token)

        print(f"Inspecting {input_path}...")
        print(f"Longest token {longest_token}\n")

        with open(input_path, "rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                pattern = re.compile(search_sequence)

                positions = [match.start() for match in pattern.finditer(mm)]

                for index in positions[:3]:
                    print(f"Sequence found at byte offset: {index}")

                    start_pos = max(0, index - WINDOW_SIZE)
                    end_pos = min(len(mm), index + len(search_sequence) + WINDOW_SIZE)

                    context_bytes = mm[start_pos:end_pos]

                    decoded_context = context_bytes.decode("utf-8", errors="ignore")

                    print("--- Decoded Context Around Match ---")
                    print(decoded_context)
                    print("------------------------------------")
