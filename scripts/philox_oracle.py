"""Generate the Philox gate fixture of `pkpdutils-core`.

`numpy.random.Philox` is Random123's philox4x64-10, the same generator the
kernel draws its bootstrap numbers from, so numpy is an independent
implementation of the algorithm rather than a second copy of ours. The stream
of the kernel is `block(counter = i >> 2)[i & 3]` with the key `[seed, 0]`; this
script writes numpy's answer for it and `crates/core/tests/philox_oracle.rs`
compares the kernel's against it.

numpy increments its counter before it generates, so `counter=c` yields block
`c + 1` and its first four raw words are block 1 and not block 0. The block of
a counter `c` is therefore `Philox(counter=c - 1).random_raw(4)`, and block 0 is
reached with the counter `2**256 - 1`, which wraps to 0 (measured, not assumed:
the all ones counter reproduces the block the round function gives for 0). A
long stream is block 0 followed by `random_raw` of a counter of 0, which is
blocks 1 and up in order.

Two sections per seed, so that the file stays small and the gate stays long:

* `words`: the first 32 words verbatim, which is what a human reads when the
  test fails and what pins the word order inside a block.
* `digest`: the FNV-1a 64 hash of the little-endian bytes of the first 100 000
  words, which pins the whole stream, including the counter carry every fourth
  word, at 8 bytes in the file.

Usage: python scripts/philox_oracle.py [--check]
"""

import json
import sys
from pathlib import Path

import numpy as np
from numpy.random import Philox

SEEDS = (0, 1, 12345, 2**64 - 1)
VERBATIM = 32
DIGEST_WORDS = 100_000
FIXTURE = (
    Path(__file__).resolve().parent.parent / "crates/core/tests/data/philox_oracle.json"
)

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3


def fnv1a(words: np.ndarray) -> int:
    """The FNV-1a 64 hash of the little-endian bytes of every word."""
    h = FNV_OFFSET
    for byte in words.astype("<u8").tobytes():
        h = ((h ^ byte) * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def stream(seed: int, n: int) -> np.ndarray:
    """The first `n` words of the kernel's stream for `seed`, from numpy.

    Word `i` is word `i & 3` of block `i >> 2`. Block 0 comes from the counter
    which wraps to it and the blocks above it from the counter numpy starts at,
    see the module docstring.
    """
    key = np.array([seed, 0], dtype=np.uint64)
    wrapped = np.array([2**64 - 1] * 4, dtype=np.uint64)
    first = Philox(key=key, counter=wrapped).random_raw(4)
    if n <= 4:
        return first[:n]
    rest = Philox(key=key, counter=0).random_raw(n - 4)
    return np.concatenate([first, rest])


def fixture() -> dict[str, object]:
    """The whole fixture, numpy's answer for every seed."""
    return {
        "source": "numpy.random.Philox, which is Random123 philox4x64-10",
        "numpy": np.__version__,
        "recipe": "word i = block(counter = i >> 2)[i & 3], key = [seed, 0]; numpy's counter is pre-incremented, so block 0 is its wrapped counter",
        "digest": f"FNV-1a 64 over the little-endian bytes of the first {DIGEST_WORDS} words",
        "seeds": [
            {
                "seed": str(seed),
                "words": [f"{int(w):016x}" for w in stream(seed, VERBATIM)],
                "digest": f"{fnv1a(stream(seed, DIGEST_WORDS)):016x}",
            }
            for seed in SEEDS
        ],
    }


def main() -> int:
    """Write the fixture, or compare it with `--check`."""
    new = fixture()
    text = json.dumps(new, indent=2) + "\n"
    if "--check" in sys.argv:
        old = FIXTURE.read_text()
        if old != text:
            print(f"{FIXTURE} is stale, run python scripts/philox_oracle.py")
            return 1
        print(f"{FIXTURE} is what numpy {np.__version__} produces")
        return 0
    FIXTURE.write_text(text)
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
