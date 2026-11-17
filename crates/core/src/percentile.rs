//! The NaN-aware percentile of the rows of a matrix.
//!
//! The port of `pkpdutils.result.nan_percentile`, which is `numpy.nanpercentile(values, q,
//! axis=-1)` with the default linear interpolation, bit for bit: the sorted finite values, the
//! virtual index `(k - 1) * (q / 100)` of the finite count `k` (the division comes first, as in
//! numpy), and numpy's two-branch interpolation, which anchors at the nearer of the two order
//! statistics (`low + span * gamma` below `gamma = 0.5`, `high - span * (1 - gamma)` from it).
//! Neither `(1 - gamma) * low + gamma * high` nor the one-branch form gives the same last bits.
//! An infinite value is an ordinary value, so a span of `inf - inf` is `NaN`, as in numpy.

/// The percentile `q` (in `[0, 100]`) of every row of the row-major `(n_rows, n_cols)` slice
/// `values`, ignoring `NaN`; a row without a finite or infinite value is `NaN`.
///
/// A slice whose length is no multiple of `n_cols`, or `n_cols == 0`, has no complete row and
/// gives an empty result for `n_cols == 0` and the complete rows otherwise.
// the summary statistics and the bootstrap of the later tasks of the plan are its callers
#[allow(dead_code)]
pub(crate) fn nan_percentile_rows(values: &[f64], n_cols: usize, q: f64) -> Vec<f64> {
    if n_cols == 0 {
        return Vec::new();
    }
    let level = q / 100.0;
    let mut sorted = Vec::with_capacity(n_cols);
    values
        .chunks_exact(n_cols)
        .map(|row| {
            sorted.clear();
            sorted.extend(row.iter().copied().filter(|x| !x.is_nan()));
            sorted.sort_by(f64::total_cmp);
            percentile_of_sorted(&sorted, level)
        })
        .collect()
}

/// The linear percentile at `level = q / 100` of the ascending values without `NaN`.
fn percentile_of_sorted(sorted: &[f64], level: f64) -> f64 {
    let count = sorted.len();
    if count == 0 || level.is_nan() {
        return f64::NAN;
    }
    let last = count - 1;
    let virtual_index = last as f64 * level;
    // an index outside the values takes the end value on both sides and still goes through the
    // interpolation, where the span of an infinite end value is `inf - inf = NaN`, as in numpy
    let (below, above) = if virtual_index >= last as f64 {
        (last, last)
    } else if virtual_index < 0.0 {
        (0, 0)
    } else {
        let below = virtual_index.floor() as usize;
        (below, below + 1)
    };
    let gamma = virtual_index - below as f64;
    let (low, high) = (sorted[below], sorted[above]);
    let span = high - low;
    if gamma >= 0.5 {
        high - span * (1.0 - gamma)
    } else {
        low + span * gamma
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Case = (&'static [f64], f64, f64);

    fn check(cases: &[Case]) {
        for (row, q, expected) in cases {
            let got = nan_percentile_rows(row, row.len(), *q)[0];
            assert_eq!(got.to_bits(), expected.to_bits(), "{row:?} at q = {q}");
        }
    }

    // the tables are generated once and committed, by
    //   python scripts/reference_percentile.py plain
    // with numpy 2.5.4 (`numpy.nanpercentile(row, q)`), never by this code
    const PLAIN: &[Case] = &[
        (
            &[
                1.00123091030648,
                1.3481665220093473,
                0.7602272666316231,
                0.4104127826686174,
                0.6346568720123182,
            ],
            0.0,
            0.4104127826686174,
        ),
        (
            &[
                0.3709653729873708,
                1.0619890399804632,
                3.8198656259791655,
                0.6112761129387685,
                0.5376890285297825,
            ],
            10.0,
            0.4376548352043355,
        ),
        (
            &[
                1.6320584162712972,
                1.4288744095836212,
                1.1111708164263712,
                0.3943690848052393,
                0.9711718707780436,
            ],
            25.0,
            0.9711718707780436,
        ),
        (
            &[
                2.004316679535085,
                0.26074442987006247,
                0.6327905722921364,
                0.14938584748286632,
                0.2753980592362162,
            ],
            50.0,
            0.2753980592362162,
        ),
        (
            &[
                0.15854211078095357,
                0.7904988073402927,
                0.2815496468908558,
                1.311621763340773,
                1.169704422585226,
            ],
            75.0,
            1.169704422585226,
        ),
        (
            &[
                0.8295010159072586,
                0.08072074199103559,
                0.5835104630708957,
                0.9526564386629909,
                1.1199779369971745,
            ],
            90.0,
            1.053049337663501,
        ),
        (
            &[
                0.21650627123723007,
                0.6201751901902539,
                0.37586731704948784,
                0.445375630486897,
                2.888965915495888,
            ],
            100.0,
            2.888965915495888,
        ),
        (
            &[
                0.44595613878393364,
                0.968001439184788,
                2.4215065005788143,
                0.5578861148157961,
                0.8943107674778168,
                1.1167963036477326,
                1.065859775336746,
                0.2937413013412064,
            ],
            0.0,
            0.2937413013412064,
        ),
        (
            &[
                1.0791138880949929,
                3.891611821229294,
                0.21285487614699775,
                2.361702336479686,
                1.1267687527779464,
                0.5265176661092926,
                7.392134624352356,
                2.1431135725657775,
            ],
            10.0,
            0.43241882912060414,
        ),
        (
            &[
                0.3014084666512304,
                1.0773628276554947,
                1.780135675642311,
                0.8279668800382102,
                1.9796306109886617,
                0.9356467101751683,
                1.9488657970826928,
                4.214464729843154,
            ],
            25.0,
            0.9087267526409288,
        ),
        (
            &[
                0.5088193428833176,
                1.2252422877887268,
                0.6291990758692184,
                1.1357218173590427,
                0.30507594668400106,
                0.5602895381246513,
                0.8218511584190247,
                2.4565646055070345,
            ],
            50.0,
            0.7255251171441215,
        ),
        (
            &[
                3.143139079691529,
                0.26619456439441125,
                0.451742764614345,
                1.909618382911962,
                0.13636505189585313,
                0.6292857298383873,
                0.907295642372379,
                3.5149137149240994,
            ],
            75.0,
            2.2179985571068537,
        ),
        (
            &[
                1.9925274346992534,
                0.720929865594809,
                0.6917187102370075,
                0.7786486198619015,
                4.5883909197084725,
                0.6517951571477772,
                0.7380967330595681,
                1.4227463702623067,
            ],
            90.0,
            2.7712864802020185,
        ),
        (
            &[
                0.886237376388637,
                0.8209572611365016,
                0.32822131963728884,
                0.988544649905462,
                0.641734108064361,
                3.2095404862380357,
                1.9214661274786087,
                0.9761455124958935,
            ],
            100.0,
            3.2095404862380357,
        ),
    ];

    // rows where the one-branch form `low + span * gamma` misses numpy in the last bit, by
    //   python scripts/reference_percentile.py single
    const ONE_BRANCH_DIFFERS: &[Case] = &[
        (
            &[
                0.4065990933599784,
                0.4352985413172821,
                0.7375090809352416,
                1.178276521398737,
                9.438118288911602,
            ],
            95.0,
            7.786149935409028,
        ),
        (
            &[0.11900770479118408, 0.322578916445936, 1.437405567257257],
            75.0,
            0.8799922418515965,
        ),
        (
            &[
                0.11075711010963588,
                0.7477274937154043,
                0.9208313533660131,
                0.9787563272806166,
                1.0844298558611771,
                3.1726580781283706,
            ],
            10.0,
            0.42924230191252005,
        ),
        (
            &[1.3525287739008662, 1.366923828787326, 3.1113156617474664],
            90.0,
            2.7624372951554386,
        ),
    ];

    // rows where `(1 - gamma) * low + gamma * high` misses numpy in the last bit, by
    //   python scripts/reference_percentile.py naive
    const NAIVE_DIFFERS: &[Case] = &[
        (
            &[
                0.5088193428833176,
                0.9356467101751683,
                1.2252422877887268,
                1.9488657970826928,
                4.214464729843154,
            ],
            90.0,
            3.30822515673897,
        ),
        (
            &[
                0.7394359563772827,
                0.859466313406796,
                1.0224702950269948,
                3.243030785707802,
            ],
            10.0,
            0.7754450634861366,
        ),
        (
            &[
                0.3682316602880776,
                0.412037107451691,
                0.45704445535320487,
                0.5178556249745192,
                1.2158068345378976,
                1.4046047204126453,
                1.4277021858207057,
            ],
            90.0,
            1.4138437065758696,
        ),
        (
            &[
                0.26589718947552565,
                0.28053667280592787,
                0.5044699131205532,
                0.8714289213947561,
                0.9926687196176343,
                1.6671386353866224,
                2.985734188507344,
                5.595550027242015,
            ],
            33.3,
            0.6259333448593142,
        ),
    ];

    // rows with three holes among seven, by
    //   python scripts/reference_percentile.py hole
    const HOLES: &[Case] = &[
        (
            &[
                f64::NAN,
                1.3481665220093473,
                f64::NAN,
                0.4104127826686174,
                f64::NAN,
                0.3709653729873708,
                1.0619890399804632,
            ],
            50.0,
            0.7362009113245404,
        ),
        (
            &[
                1.6320584162712972,
                1.4288744095836212,
                1.1111708164263712,
                0.3943690848052393,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
            75.0,
            1.4796704112555403,
        ),
        (
            &[
                0.15854211078095357,
                0.7904988073402927,
                0.2815496468908558,
                1.311621763340773,
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ],
            50.0,
            0.5360242271155742,
        ),
        (
            &[
                0.21650627123723007,
                0.6201751901902539,
                0.37586731704948784,
                f64::NAN,
                f64::NAN,
                0.44595613878393364,
                f64::NAN,
            ],
            25.0,
            0.3360270555964234,
        ),
        (
            &[
                1.1167963036477326,
                1.065859775336746,
                0.2937413013412064,
                f64::NAN,
                f64::NAN,
                f64::NAN,
                2.361702336479686,
            ],
            50.0,
            1.0913280394922393,
        ),
        (
            &[
                2.1431135725657775,
                0.3014084666512304,
                f64::NAN,
                1.780135675642311,
                0.8279668800382102,
                f64::NAN,
                f64::NAN,
            ],
            50.0,
            1.3040512778402606,
        ),
    ];

    #[test]
    fn the_percentile_matches_numpy_on_a_row_without_holes() {
        check(PLAIN);
    }

    #[test]
    fn the_interpolation_is_the_two_branch_form_of_numpy() {
        check(ONE_BRANCH_DIFFERS);
        check(NAIVE_DIFFERS);
        // the table really holds the cases it is named for
        for (row, q, expected) in ONE_BRANCH_DIFFERS.iter().chain(NAIVE_DIFFERS) {
            let index = (row.len() - 1) as f64 * (q / 100.0);
            let b = index.floor() as usize;
            let g = index - b as f64;
            let (lo, hi) = (row[b], row[b + 1]);
            let one = lo + (hi - lo) * g;
            let naive = (1.0 - g) * lo + g * hi;
            assert!(one != *expected || naive != *expected, "{row:?}");
        }
        assert!(ONE_BRANCH_DIFFERS.iter().all(|(row, q, e)| {
            let index = (row.len() - 1) as f64 * (q / 100.0);
            let b = index.floor() as usize;
            let g = index - b as f64;
            row[b] + (row[b + 1] - row[b]) * g != *e
        }));
        assert!(NAIVE_DIFFERS.iter().all(|(row, q, e)| {
            let index = (row.len() - 1) as f64 * (q / 100.0);
            let b = index.floor() as usize;
            let g = index - b as f64;
            (1.0 - g) * row[b] + g * row[b + 1] != *e
        }));
    }

    #[test]
    fn a_hole_shifts_the_virtual_index_to_the_finite_count() {
        let nan = f64::NAN;
        // the median of the three finite values, not of four
        assert_eq!(nan_percentile_rows(&[1.0, 2.0, nan, 4.0], 4, 50.0), [2.0]);
        // 25 of [1, 2, 4] sits at index 0.5
        assert_eq!(
            nan_percentile_rows(&[nan, 1.0, 2.0, nan, 4.0], 5, 25.0),
            [1.5]
        );
        check(HOLES);
    }

    #[test]
    fn a_row_of_only_nan_is_nan_and_a_row_of_one_value_is_that_value() {
        let nan = f64::NAN;
        let out = nan_percentile_rows(&[nan, nan, nan, 7.5, nan, nan], 3, 50.0);
        assert!(out[0].is_nan());
        assert_eq!(out[1], 7.5);
        for q in [0.0, 25.0, 100.0] {
            assert_eq!(nan_percentile_rows(&[nan, 7.5], 2, q), [7.5]);
        }
    }

    #[test]
    fn rows_are_reduced_independently() {
        let values = [3.0, 1.0, 2.0, 10.0, f64::NAN, 30.0];
        assert_eq!(nan_percentile_rows(&values, 3, 50.0), [2.0, 20.0]);
        assert!(nan_percentile_rows(&values, 0, 50.0).is_empty());
        assert!(nan_percentile_rows(&[], 3, 50.0).is_empty());
    }

    #[test]
    fn an_infinity_is_an_ordinary_value() {
        let inf = f64::INFINITY;
        // numpy.nanpercentile([1, inf], 25) is inf; at 50 it is nan (inf - inf), with a warning
        assert_eq!(nan_percentile_rows(&[1.0, inf], 2, 25.0), [inf]);
        assert!(nan_percentile_rows(&[1.0, inf], 2, 50.0)[0].is_nan());
        assert!(nan_percentile_rows(&[-inf, inf], 2, 50.0)[0].is_nan());
        // an exact index still multiplies the infinite span by a zero gamma, as in numpy
        assert!(nan_percentile_rows(&[1.0, 2.0, inf], 3, 50.0)[0].is_nan());
        assert!(nan_percentile_rows(&[1.0, 2.0, inf], 3, 75.0)[0].is_nan());
        assert!(nan_percentile_rows(&[1.0, inf], 2, 100.0)[0].is_nan());
        assert!(nan_percentile_rows(&[-inf, 1.0], 2, 0.0)[0].is_nan());
        assert_eq!(nan_percentile_rows(&[-inf, 1.0], 2, 100.0), [1.0]);
    }
}
