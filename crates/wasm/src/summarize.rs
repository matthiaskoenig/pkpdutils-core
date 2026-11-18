//! The summary statistics of a parameter table: raw values in, one row per variable and group
//! out.

use pkpdutils_abi::SummaryAnswer;
use serde_json::{Value, json};

use crate::output::{Aligned, error, kind_of, ok_from, publish, read_request};

/// The statistics of an answer, in the order the payload carries them.
///
/// The kernel computes every statistic of every variable and a host drops the ones it does not
/// want, which is what `ParameterResult.summarize` of python does; the names here are the ones
/// of that result without its `x_` prefix.
const STATISTICS: [&str; 14] = [
    "mean", "sd", "se", "cv", "ci_low", "ci_high", "median", "q25", "q75", "min", "max", "count",
    "geomean", "geocv",
];

/// The statistics of an answer as slices, in the order of [`STATISTICS`].
fn parts(answer: &SummaryAnswer) -> [&[f64]; 14] {
    [
        &answer.mean,
        &answer.sd,
        &answer.se,
        &answer.cv,
        &answer.ci_low,
        &answer.ci_high,
        &answer.median,
        &answer.q25,
        &answer.q75,
        &answer.min,
        &answer.max,
        &answer.count,
        &answer.geomean,
        &answer.geocv,
    ]
}

/// Reduce a parameter table over its samples, the pure function the export goes through.
///
/// The request carries the values and no names (`pkpdutils_abi::SummaryRequest`), so the header
/// of the answer names the statistics and the dimensions and the page knows its own variables.
/// Every statistic is one `(n_vars, n_groups)` block of the payload, in the order the header
/// declares.
pub(crate) fn summarize(request: &[u8]) -> Option<Aligned> {
    let text = match std::str::from_utf8(request) {
        Ok(text) => text,
        Err(bad) => return error("malformed", &format!("the request is no UTF-8: {bad}")),
    };
    let request: pkpdutils_abi::SummaryRequest = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(bad) => return error("malformed", &bad.to_string()),
    };
    let answer = match pkpdutils_core::summarize(&request) {
        Ok(answer) => answer,
        Err(bad) => return error(kind_of(&bad), &bad.to_string()),
    };
    let header = header_of(&request);
    ok_from(&header, &parts(&answer))
}

/// The header of an answer: the statistics of the payload and the shape of every block.
fn header_of(request: &pkpdutils_abi::SummaryRequest) -> String {
    let text: Value = json!({
        "statistics": STATISTICS,
        "n_vars": request.n_vars,
        "n_groups": request.n_groups,
        "n_samples": request.n_samples,
        "width": request.n_vars * request.n_groups,
    });
    text.to_string()
}

/// Reduce a parameter table: a JSON request of `len` bytes at `ptr`, the answer in the output.
///
/// # Safety
///
/// `ptr` is an address `pkpd_alloc` returned and `len` the number of bytes written into it.
///
/// Returns 0 when the output is an answer and 1 when it is an error header.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pkpd_summarize(ptr: usize, len: usize) -> i32 {
    crate::output::install_panic_hook();
    // SAFETY: the caller guarantees the range is the request it wrote.
    let request = unsafe { read_request(ptr, len) };
    let out = summarize(request);
    let failed = i32::from(crate::output::is_error(out.as_ref()));
    publish(out);
    failed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tests::{header_text, kind_text, payload};

    /// Two variables of one group of three samples.
    fn request() -> Vec<u8> {
        let request = pkpdutils_abi::SummaryRequest {
            values: vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            n_vars: 2,
            n_groups: 1,
            n_samples: 3,
            ci_level: 0.95,
            lognormal: vec![false, true],
            discrete: vec![false, false],
        };
        serde_json::to_vec(&request).unwrap()
    }

    /// One statistic of an answer, by name.
    fn statistic(out: &Aligned, name: &str) -> Vec<f64> {
        let header: Value = serde_json::from_str(&header_text(out)).unwrap();
        let width = header["width"].as_u64().unwrap() as usize;
        let at = header["statistics"]
            .as_array()
            .unwrap()
            .iter()
            .position(|statistic| statistic == name)
            .unwrap();
        payload(out)[at * width..(at + 1) * width].to_vec()
    }

    #[test]
    fn a_summary_request_answers_one_block_per_statistic() {
        let out = summarize(&request()).unwrap();
        let header: Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["statistics"], json!(STATISTICS));
        assert_eq!(header["n_vars"], 2);
        assert_eq!(header["n_groups"], 1);
        assert_eq!(header["width"], 2);
        assert_eq!(payload(&out).len(), STATISTICS.len() * 2);
        assert_eq!(statistic(&out, "mean"), vec![2.0, 20.0]);
        assert_eq!(statistic(&out, "count"), vec![3.0, 3.0]);
        assert_eq!(statistic(&out, "min"), vec![1.0, 10.0]);
        // the geometric mean of 1, 2, 3 is the cube root of 6
        let geomean = statistic(&out, "geomean")[0];
        assert!((geomean - 6.0f64.cbrt()).abs() < 1e-12, "{geomean}");
    }

    #[test]
    fn a_malformed_summary_request_is_malformed_and_never_refused() {
        // the lengths of the request contradict its dimensions, which is python's own error
        let mut wire: Value = serde_json::from_slice(&request()).unwrap();
        wire["n_samples"] = json!(4);
        let out = summarize(wire.to_string().as_bytes()).unwrap();
        assert_eq!(kind_text(&out), "malformed");
        let header: Value = serde_json::from_str(&header_text(&out)).unwrap();
        let message = header["error"]["message"].as_str().unwrap();
        assert!(message.contains("6 values"), "{message}");

        // and a request which is not a summary request at all is malformed as well
        let out = summarize(br#"{"values":[1.0]}"#).unwrap();
        assert_eq!(kind_text(&out), "malformed");
    }

    #[test]
    fn a_missing_value_is_left_out_of_the_statistics() {
        let mut wire: Value = serde_json::from_slice(&request()).unwrap();
        wire["values"] = json!([1.0, "NaN", 3.0, 10.0, 20.0, 30.0]);
        let out = summarize(wire.to_string().as_bytes()).unwrap();
        assert_eq!(statistic(&out, "count"), vec![2.0, 3.0]);
        assert_eq!(statistic(&out, "mean"), vec![2.0, 20.0]);
    }
}
