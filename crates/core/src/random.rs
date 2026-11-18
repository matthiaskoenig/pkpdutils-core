//! The random numbers of the bootstrap, as a pure function of the seed and the index.
//!
//! A sequential generator cannot serve this kernel. The rows of a batch run in shards
//! ([`crate::shard`]), so a shard has to reach the draws of its own rows without drawing the rows
//! before it, and a host whose own engine analyses a batch the kernel refuses has to be able to
//! ask for the draws of any index range. Both need random access by index, which is what a
//! counter-based generator gives: there is no state to advance, and the draw of an index is a
//! function of that index alone.
//!
//! The generator is Philox4x64-10 of Random123. It was chosen over the 4x32 variant for its
//! oracle rather than for its width: `numpy.random.Philox` is philox4x64-10, so the stream of
//! this module is checked against an independent implementation of the same algorithm which is
//! installed next to the python package (`scripts/philox_oracle.py`,
//! `crates/core/tests/philox_oracle.rs`) instead of against a handful of test vectors copied out
//! of a paper. It needs no entropy, no clock and no allocation, which is what makes it build for
//! `wasm32-unknown-unknown`, and it is integer arithmetic throughout, so its stream is bit
//! identical on every target.
//!
//! The normal transform is the inverse cumulative distribution function and this is not an
//! implementation detail: a ziggurat or a rejection transform consumes a variable number of
//! uniforms per normal, which would make the position in the stream depend on the values drawn
//! before it, and the index would stop being a position. One uniform in, one normal out.
//!
//! Reproducing `numpy.random.Generator.standard_normal` is explicitly not a goal. numpy promises
//! its stream only on the same build, in the same environment, on the same machine, and reserves
//! the right to change a distribution method; this module promises the same numbers on every
//! platform, in the browser and across numpy releases, which is the stronger guarantee and the
//! reason the host asks the kernel for its draws rather than drawing its own.

use crate::stats::normal_ppf;

/// The number of Philox rounds, which is part of the generator's identity and not a tunable.
const ROUNDS: usize = 10;

/// The multiplier of the first and third counter words.
const MULTIPLIER_0: u64 = 0xD2E7_470E_E14C_6C93;

/// The multiplier of the second and fourth counter words.
const MULTIPLIER_1: u64 = 0xCA5A_8263_9512_1157;

/// The round bump of the first key word, the golden ratio.
const WEYL_0: u64 = 0x9E37_79B9_7F4A_7C15;

/// The round bump of the second key word, the square root of three.
const WEYL_1: u64 = 0xBB67_AE85_84CA_A73B;

/// The high and the low word of the 128 bit product.
fn mul_hilo(a: u64, b: u64) -> (u64, u64) {
    let product = u128::from(a) * u128::from(b);
    ((product >> 64) as u64, product as u64)
}

/// One Philox round of the counter under the key.
fn round(counter: [u64; 4], key: [u64; 2]) -> [u64; 4] {
    let (hi0, lo0) = mul_hilo(MULTIPLIER_0, counter[0]);
    let (hi1, lo1) = mul_hilo(MULTIPLIER_1, counter[2]);
    [
        hi1 ^ counter[1] ^ key[0],
        lo1,
        hi0 ^ counter[3] ^ key[1],
        lo0,
    ]
}

/// The four words of the block of `counter` under the key of `seed`.
///
/// The key is `[seed, 0]` and the counter is `[counter, 0, 0, 0]`, which is the mapping
/// `numpy.random.Philox(key=[seed, 0], counter=counter)` uses, so the oracle of the gate is
/// numpy's own stream.
fn block(counter: u64, seed: u64) -> [u64; 4] {
    let mut state = [counter, 0, 0, 0];
    let mut key = [seed, 0];
    for r in 0..ROUNDS {
        if r > 0 {
            key = [key[0].wrapping_add(WEYL_0), key[1].wrapping_add(WEYL_1)];
        }
        state = round(state, key);
    }
    state
}

/// The draws of one seed, reachable by index.
///
/// The index is the flat position in the C order of `(n_rows, n_boot, n_time)`, the order
/// `pkpdutils.nca.uncertainty.resample_values` draws in, and a point whose spread is not usable
/// still consumes its draw, because python draws for every point and masks afterwards. Matching
/// the mask rather than the draw would shift every later number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Draws {
    /// The seed, which is the whole state: a seed and an index determine a draw.
    seed: u64,
}

impl Draws {
    /// The draws of `seed`.
    ///
    /// Every value of `u64` is a seed, `0` included: the kernel never reads entropy and a host
    /// which wants an arbitrary seed draws it from its own and passes it.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// The raw word of `index`, which is word `index & 3` of the block of `index >> 2`.
    ///
    /// Every word of every block is used, so four consecutive indices cost one block, and
    /// `index >> 2` of a `u64` is a `u64`, so the counter never carries into its upper words and
    /// there is no largest index to document.
    #[must_use]
    pub fn word(&self, index: u64) -> u64 {
        block(index >> 2, self.seed)[(index & 3) as usize]
    }

    /// The uniform of `index`, strictly inside `(0, 1)`.
    ///
    /// The 52 bits of the word's high end carry the value and the half step opens the interval at
    /// both ends: the draws are the midpoints of the `2^52` cells of the unit interval, so the
    /// smallest is `2^-53` and the largest `1 - 2^-53`, both of them exactly representable. The
    /// half step is not cosmetic: `0.0` would make the inverse cumulative distribution function
    /// `-inf`, and one such draw would turn a whole parameter of a row into `NaN`. 52 bits rather
    /// than 53 is what makes the two ends exact: `2^53 - 1 + 0.5` is not a `f64` and rounds to
    /// `2^53`, which would make the largest uniform exactly 1.0.
    #[must_use]
    pub fn uniform(&self, index: u64) -> f64 {
        uniform_of(self.word(index))
    }

    /// The standard normal draw of `index`.
    #[must_use]
    pub fn normal(&self, index: u64) -> f64 {
        normal_of(self.word(index))
    }

    /// Write the normals of the indices `start .. start + out.len()` into `out`.
    ///
    /// One block serves four consecutive indices, so a run of draws costs a quarter of the
    /// blocks. The answer is the same as `normal` of every index on its own, whatever the range
    /// is cut into, which is what lets a host stream the draws of a large batch in chunks and a
    /// shard draw for its own rows alone.
    pub fn normals_into(&self, start: u64, out: &mut [f64]) {
        let mut written = 0usize;
        let mut index = start;
        while written < out.len() {
            let word_in_block = (index & 3) as usize;
            let words = block(index >> 2, self.seed);
            for word in words.iter().skip(word_in_block) {
                if written == out.len() {
                    break;
                }
                out[written] = normal_of(*word);
                written += 1;
                index = index.wrapping_add(1);
            }
        }
    }
}

/// The uniform of one raw word, see [`Draws::uniform`].
fn uniform_of(word: u64) -> f64 {
    // `f64::EPSILON` is `2^-52`, the step of the 52 bit grid the 52 kept bits index
    ((word >> 12) as f64 + 0.5) * f64::EPSILON
}

/// The normal of one raw word, the inverse cumulative distribution function of its uniform.
///
/// The quantile is evaluated on the smaller tail, so a draw deep in a tail keeps the precision
/// the quantile has there. `1.0 - u` is exact for every uniform this module produces, since both
/// are midpoints of the `2^52` grid of `(0, 1)`, so the reflection costs no accuracy.
fn normal_of(word: u64) -> f64 {
    let u = uniform_of(word);
    if u <= 0.5 {
        normal_ppf(u)
    } else {
        -normal_ppf(1.0 - u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first words of the stream of a seed, as `scripts/philox_oracle.py` writes them.
    ///
    /// The whole stream of four seeds is compared against numpy in
    /// `crates/core/tests/philox_oracle.rs`; this is the one in-module case, so that a change of
    /// the word order inside a block fails the unit tests as well.
    #[test]
    fn the_first_words_of_the_stream_of_seed_zero() {
        let draws = Draws::new(0);
        let words: Vec<String> = (0..4).map(|i| format!("{:016x}", draws.word(i))).collect();
        assert_eq!(
            words,
            vec![
                "16554d9eca36314c",
                "db20fe9d672d0fdc",
                "d7e772cee186176b",
                "7e68b68aec7ba23b"
            ]
        );
    }

    #[test]
    fn a_draw_is_a_pure_function_of_its_index() {
        let draws = Draws::new(7);
        let mut whole = [0.0f64; 32];
        draws.normals_into(0, &mut whole);

        // the same range in three pieces, none of them a multiple of the block
        let mut pieces = Vec::new();
        for (start, count) in [(0u64, 13usize), (13, 7), (20, 12)] {
            let mut part = vec![0.0; count];
            draws.normals_into(start, &mut part);
            pieces.extend(part);
        }
        assert_eq!(pieces, whole.to_vec());

        // and one index at a time, which is what a shard of one row does
        for (i, value) in whole.iter().enumerate() {
            assert_eq!(draws.normal(i as u64), *value, "index {i}");
        }

        // a range which starts inside a block and is shorter than the rest of it
        let mut short = [0.0f64; 2];
        draws.normals_into(5, &mut short);
        assert_eq!(short.to_vec(), whole[5..7].to_vec());

        // an empty range writes nothing and does not spin
        let mut empty: [f64; 0] = [];
        draws.normals_into(11, &mut empty);
    }

    #[test]
    fn a_seed_is_the_whole_state_and_zero_is_a_seed() {
        let a = Draws::new(0);
        let b = Draws::new(1);
        assert_eq!(a.normal(3), Draws::new(0).normal(3));
        assert_ne!(a.normal(3), b.normal(3));
        // the seeds of the far end of the range are seeds as well
        assert_ne!(Draws::new(u64::MAX).normal(0), a.normal(0));
    }

    #[test]
    fn the_uniform_is_strictly_inside_the_unit_interval() {
        // the two ends of the word range, where a uniform would reach 0 or 1 if it could
        assert!(uniform_of(0) > 0.0, "{}", uniform_of(0));
        assert!(uniform_of(u64::MAX) < 1.0, "{}", uniform_of(u64::MAX));
        assert!(normal_of(0).is_finite(), "{}", normal_of(0));
        assert!(normal_of(u64::MAX).is_finite(), "{}", normal_of(u64::MAX));

        let draws = Draws::new(99);
        for index in (0..100_000).chain(u64::MAX - 100_000..u64::MAX) {
            let u = draws.uniform(index);
            assert!(u > 0.0 && u < 1.0, "index {index} gave {u}");
            assert!(draws.normal(index).is_finite(), "index {index}");
        }
    }

    #[test]
    fn the_normals_have_the_moments_of_a_normal() {
        let draws = Draws::new(2024);
        let n = 200_000;
        let mut values = vec![0.0; n];
        draws.normals_into(0, &mut values);

        let mean = values.iter().sum::<f64>() / n as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        assert!(mean.abs() < 0.01, "mean {mean}");
        assert!(
            (variance.sqrt() - 1.0).abs() < 0.01,
            "sd {}",
            variance.sqrt()
        );

        for (bound, share) in [(1.0, 0.682_689), (2.0, 0.954_500), (3.0, 0.997_300)] {
            let inside = values.iter().filter(|v| v.abs() < bound).count() as f64 / n as f64;
            assert!(
                (inside - share).abs() < 0.005,
                "within {bound} sigma: {inside}, expected about {share}"
            );
        }

        // the transform is symmetric, so the tails hold about the same count
        let low = values.iter().filter(|v| **v < -2.0).count();
        let high = values.iter().filter(|v| **v > 2.0).count();
        let imbalance = (low as f64 - high as f64).abs() / (low + high) as f64;
        assert!(imbalance < 0.05, "tails {low} against {high}");
    }
}
