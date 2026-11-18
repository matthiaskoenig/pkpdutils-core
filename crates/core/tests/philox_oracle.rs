//! The integer stream of the kernel's generator against numpy's.
//!
//! `numpy.random.Philox` is Random123 philox4x64-10, the generator of
//! `pkpdutils_core::Draws`, so numpy is an independent implementation of the same algorithm and
//! not a second copy of this one. `scripts/philox_oracle.py` writes its answer into
//! `data/philox_oracle.json` and this test compares the kernel's against it: the first 32 words
//! of four seeds verbatim, which is what a human reads when it fails, and the FNV-1a 64 hash of
//! the first 100 000 words, which pins the whole stream, the block boundary every fourth word
//! included, in eight bytes of fixture.
//!
//! Regenerate the fixture with `python scripts/philox_oracle.py` and check it with `--check`.
//! Never edit it by hand: it is numpy's answer and not ours.

use pkpdutils_core::Draws;
use serde::Deserialize;

/// The fixture as `scripts/philox_oracle.py` writes it.
#[derive(Debug, Deserialize)]
struct Fixture {
    /// Where the numbers come from.
    source: String,
    /// The numpy release which produced them.
    numpy: String,
    /// One entry per seed.
    seeds: Vec<SeedCase>,
}

/// Numpy's answer for one seed.
#[derive(Debug, Deserialize)]
struct SeedCase {
    /// The seed, as a string because it does not fit a JSON number near `u64::MAX`.
    seed: String,
    /// The first words of the stream, lower case hexadecimal of sixteen digits.
    words: Vec<String>,
    /// The digest of the first 100 000 words.
    digest: String,
}

/// The number of words the digest covers, which the fixture's own header states as well.
const DIGEST_WORDS: u64 = 100_000;

/// The FNV-1a 64 hash of the little-endian bytes of every word, as the script computes it.
fn fnv1a(words: impl Iterator<Item = u64>) -> u64 {
    let mut hash = 0xCBF2_9CE4_8422_2325u64;
    for word in words {
        for byte in word.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    hash
}

#[test]
fn the_integer_stream_is_the_one_numpy_produces() {
    let fixture: Fixture =
        serde_json::from_str(include_str!("data/philox_oracle.json")).expect("fixture parses");
    assert!(
        fixture.source.contains("philox4x64-10"),
        "the fixture names its generator: {}",
        fixture.source
    );
    assert!(!fixture.numpy.is_empty(), "the fixture names its numpy");
    assert_eq!(fixture.seeds.len(), 4, "four seeds are compared");

    for case in &fixture.seeds {
        let seed: u64 = case.seed.parse().expect("the seed parses");
        let draws = Draws::new(seed);

        let words: Vec<String> = (0..case.words.len() as u64)
            .map(|index| format!("{:016x}", draws.word(index)))
            .collect();
        assert_eq!(words, case.words, "the first words of the seed {seed}");

        let digest = fnv1a((0..DIGEST_WORDS).map(|index| draws.word(index)));
        assert_eq!(
            format!("{digest:016x}"),
            case.digest,
            "the digest of {DIGEST_WORDS} words of the seed {seed}"
        );
    }
}
