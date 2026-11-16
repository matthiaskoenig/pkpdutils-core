//! Packing the valid points of one curve to the front of its row.
//!
//! The python kernel works on `(N, n)` arrays and keeps every row the same width, so
//! `pack_valid` (`src/pkpdutils/nca/auc.py`) moves the points whose time and value are both
//! finite to the front of the row with a stable `argsort` over the validity flag and leaves
//! `NaN` behind them. Every later step then reads `n_valid` points and ignores the rest. Here a
//! row is a pair of slices and the count comes back from the call, but the `NaN` padding is
//! written all the same: a caller which does look past the count reads a `NaN` and not a stale
//! value out of the buffer it handed over.
//!
//! What the packing deliberately does not do is sort, and it does not drop a pre-dose sample
//! with a negative time. A stable partition over `~valid` is not a sort by time, and the
//! trapezoid rules that follow assume a row which already arrives in time order. Where
//! `pkpdutils` settles that ordering is not recorded (`areas.md` trap T1); the kernel does not,
//! and neither does this.

/// Move the valid points of one row to the front of `t` and `c`, keeping their order.
///
/// A point is valid when its time and its value are both finite, so a `NaN` and an infinity are
/// both invalid, as `numpy.isfinite` has it. The slots behind the points written are filled
/// with `NaN`, which is the padding of the python arrays.
///
/// The shapes come from a host, so none of them is trusted: the points read are the ones
/// `times` and `values` both have, the points written are the ones `t` and `c` both have room
/// for, and the count returned is what was really written.
///
/// Mirrors `pack_valid` of `src/pkpdutils/nca/auc.py`.
pub(crate) fn pack_valid(times: &[f64], values: &[f64], t: &mut [f64], c: &mut [f64]) -> usize {
    let mut n_valid = 0usize;
    for (&time, &value) in times.iter().zip(values.iter()) {
        if !time.is_finite() || !value.is_finite() {
            continue;
        }
        match (t.get_mut(n_valid), c.get_mut(n_valid)) {
            (Some(packed_time), Some(packed_value)) => {
                *packed_time = time;
                *packed_value = value;
                n_valid += 1;
            }
            _ => break,
        }
    }
    for slot in t.iter_mut().skip(n_valid) {
        *slot = f64::NAN;
    }
    for slot in c.iter_mut().skip(n_valid) {
        *slot = f64::NAN;
    }
    n_valid
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `t[..n]`, `c[..n]` and the two tails, so that a test reads as the row does.
    fn packed(times: &[f64], values: &[f64]) -> (Vec<f64>, Vec<f64>, usize) {
        let width = times.len().max(values.len());
        let mut t = vec![0.0; width];
        let mut c = vec![0.0; width];
        let n = pack_valid(times, values, &mut t, &mut c);
        (t, c, n)
    }

    #[test]
    fn packing_moves_the_valid_points_to_the_front_in_order() {
        let times = [0.0, 1.0, f64::NAN, 3.0, 4.0, 5.0];
        let values = [1.0, f64::NAN, 9.0, 7.0, f64::INFINITY, 2.0];
        let (t, c, n) = packed(&times, &values);

        assert_eq!(
            n, 3,
            "a NaN time, a NaN value and an infinite value are invalid"
        );
        assert_eq!(&t[..n], &[0.0, 3.0, 5.0]);
        assert_eq!(&c[..n], &[1.0, 7.0, 2.0]);
        assert!(
            t[n..].iter().all(|x| x.is_nan()),
            "the tail of the times is NaN: {t:?}"
        );
        assert!(
            c[n..].iter().all(|x| x.is_nan()),
            "the tail of the values is NaN: {c:?}"
        );
    }

    #[test]
    fn packing_keeps_a_negative_time_and_does_not_sort() {
        // areas.md section 1 and trap T1/T2: pack_valid neither sorts nor drops a
        // pre-dose sample. The python `argsort(~valid)` is a stable partition on the
        // validity flag alone, so a row arrives in the order the host sent it.
        let (t, c, n) = packed(&[5.0, -1.0, 2.0], &[3.0, 1.0, 2.0]);

        assert_eq!(n, 3);
        assert_eq!(&t[..n], &[5.0, -1.0, 2.0], "the times keep their order");
        assert_eq!(&c[..n], &[3.0, 1.0, 2.0]);
    }

    #[test]
    fn a_row_of_one_valid_point_and_a_row_of_none_pack_without_panicking() {
        let (t, c, n) = packed(&[f64::NAN, 2.0], &[1.0, 4.0]);
        assert_eq!(n, 1);
        assert_eq!(&t[..n], &[2.0]);
        assert_eq!(&c[..n], &[4.0]);
        assert!(t[n..].iter().all(|x| x.is_nan()));

        let (t, c, n) = packed(&[f64::NAN, f64::NAN], &[1.0, 2.0]);
        assert_eq!(n, 0);
        assert!(
            t.iter().all(|x| x.is_nan()),
            "an empty row is all NaN: {t:?}"
        );
        assert!(
            c.iter().all(|x| x.is_nan()),
            "an empty row is all NaN: {c:?}"
        );

        let (_, _, n) = packed(&[], &[]);
        assert_eq!(n, 0, "a row of no points at all");
    }

    #[test]
    fn a_short_buffer_or_a_short_row_stops_at_its_end() {
        // A host controls the shapes, so neither a buffer shorter than the row nor a
        // pair of rows of different length may index out of bounds.
        let mut t = [0.0; 2];
        let mut c = [0.0; 2];
        let n = pack_valid(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0], &mut t, &mut c);
        assert_eq!(n, 2, "the buffer holds two of the three points");
        assert_eq!(t, [0.0, 1.0]);
        assert_eq!(c, [1.0, 2.0]);

        let mut t = [0.0; 3];
        let mut c = [0.0; 3];
        let n = pack_valid(&[0.0, 1.0, 2.0], &[1.0, 2.0], &mut t, &mut c);
        assert_eq!(n, 2, "a value without a time is no point");
        assert!(t[2].is_nan());
    }
}
