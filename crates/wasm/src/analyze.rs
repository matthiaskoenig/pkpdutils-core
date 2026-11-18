//! The analysis of a batch: a JSON request in, the columns of a result out.

use pkpdutils_abi::AnalysisResult;
use pkpdutils_core::KernelError;
use serde_json::{Value, json};

use crate::output::{Aligned, answer, error, kind_of, publish, read_request};

/// Analyse a batch, the pure function every test and the export go through.
///
/// The answer is the output of [`crate::output`]: a header which describes the result and
/// carries everything of it which is not a number, and a payload of every numeric column in the
/// order the header declares. An error of any kind is an error output and never a trap.
pub(crate) fn analyze(request: &[u8]) -> Option<Aligned> {
    let text = match std::str::from_utf8(request) {
        Ok(text) => text,
        Err(bad) => return error("malformed", &format!("the request is no UTF-8: {bad}")),
    };
    let request = match serde_json::from_str(text) {
        Ok(request) => request,
        Err(bad) => return error("malformed", &bad.to_string()),
    };
    match pkpdutils_core::analyze(&request) {
        Ok(result) => write(&result),
        Err(bad) => error(kind_of(&bad), &bad.to_string()),
    }
}

/// What a row which does not report a column contributes to it, for a host which merges.
///
/// A host which analyses the rows of one batch in several calls - the worker pool of the npm
/// package cuts a batch into one shard per worker - gets back one answer per shard, and two
/// shards need not report the same columns: a batch of an intravenous and an extravascular row
/// reports `cl` for the one and `cl_f` for the other. The merge has to fill the columns a shard
/// does not report exactly as the kernel fills them for a mixed batch, or the answer of a pool
/// would differ from the answer of one instance. So every column says what that value is rather
/// than leaving a host to keep a list of its own: `pkpdutils_core::parameter_fill` is where it
/// comes from, a `NaN` travels as the string of the wire encoding, and a column the lookup does
/// not know (an uncertainty companion of a parameter) is a number which is missing and therefore
/// a `NaN` as well.
fn fill_of(name: &str) -> Value {
    let fill = pkpdutils_core::parameter_fill(name).unwrap_or(f64::NAN);
    if fill.is_nan() {
        json!("NaN")
    } else {
        json!(fill)
    }
}

/// The columns of a result, in the order the payload carries them.
///
/// Every numeric variable of a result travels as `f64`: the parameters as they are, the flag
/// word of every sample and the boolean variables as numbers, because a flag word is below
/// `2^53` and a page reads one typed array rather than three. The python binding does the same
/// with them. A point variable carries `size` values per sample and names the dimension they
/// belong to. The text variables are per sample and small, so they stay in the header with
/// their values.
fn columns(result: &AnalysisResult) -> (Vec<Value>, Vec<Vec<f64>>) {
    let mut described = Vec::new();
    let mut owned = Vec::new();
    for column in &result.parameters {
        described.push(json!({
            "name": column.name,
            "unit": column.unit,
            "kind": "parameter",
            "width": 1,
            "fill": fill_of(&column.name),
        }));
    }
    described.push(json!({
        "name": "flags", "unit": "dimensionless", "kind": "flags", "width": 1, "fill": 0.0,
    }));
    owned.push(result.flags.iter().map(|bits| f64::from(*bits)).collect());
    for column in &result.boolean_variables {
        described.push(json!({
            "name": column.name, "unit": column.unit, "kind": "boolean", "width": 1, "fill": 0.0,
        }));
        owned.push(
            column
                .values
                .iter()
                .map(|yes| if *yes { 1.0 } else { 0.0 })
                .collect(),
        );
    }
    for variable in &result.point_variables {
        described.push(json!({
            "name": variable.name,
            "unit": variable.unit,
            "kind": "point",
            "dim": variable.dim,
            "width": variable.size,
            "fill": "NaN",
        }));
    }
    (described, owned)
}

/// The header of a result: everything of it which is not a number.
fn header_of(result: &AnalysisResult, described: &[Value]) -> Result<String, KernelError> {
    let n_rows: usize = result.sample_shape.iter().product();
    let text = json!({
        "contract": result.contract,
        "sample_dims": result.sample_dims,
        "sample_shape": result.sample_shape,
        "n_rows": n_rows,
        "columns": described,
        "text_variables": result.text_variables.iter().map(|column| json!({
            "name": column.name, "unit": column.unit, "values": column.values,
        })).collect::<Vec<_>>(),
        "coords": serde_json::to_value(&result.coords).map_err(internal)?,
        "attributes": serde_json::to_value(&result.attributes).map_err(internal)?,
    });
    Ok(text.to_string())
}

/// A failure of this host, which is a bug of this build and no answer to the request.
fn internal(bad: serde_json::Error) -> KernelError {
    KernelError::Internal(bad.to_string())
}

/// The output of a result.
fn write(result: &AnalysisResult) -> Option<Aligned> {
    let (described, owned) = columns(result);
    let header = match header_of(result, &described) {
        Ok(header) => header,
        Err(bad) => return error(kind_of(&bad), &bad.to_string()),
    };
    let mut parts: Vec<&[f64]> = Vec::with_capacity(described.len());
    let mut owned = owned.iter();
    for column in &result.parameters {
        parts.push(&column.values);
    }
    // the flags and the boolean variables, in the order `columns` described them
    for _ in 0..=result.boolean_variables.len() {
        parts.push(owned.next()?);
    }
    for variable in &result.point_variables {
        parts.push(&variable.values);
    }
    answer(&header, &parts)
}

/// Analyse a batch: a JSON request of `len` bytes at `ptr`, the answer in the output buffer.
///
/// # Safety
///
/// `ptr` is an address `pkpd_alloc` returned and `len` the number of bytes written into it.
///
/// Returns 0 when the output is an answer and 1 when it is an error header.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pkpd_analyze(ptr: usize, len: usize) -> i32 {
    crate::output::install_panic_hook();
    // SAFETY: the caller guarantees the range is the request it wrote.
    let request = unsafe { read_request(ptr, len) };
    let out = analyze(request);
    let failed = i32::from(crate::output::is_error(out.as_ref()));
    publish(out);
    failed
}

#[cfg(test)]
mod tests {
    use pkpdutils_abi::{
        ANALYSIS_CONTRACT, AnalysisRequest, BATCH_CONTRACT, Batch, Kind, NcaOptions, Route,
    };

    use super::*;
    use crate::output::tests::{header_text, kind_text, payload};

    /// A batch of one oral curve which halves every hour, the curve of the binding tests.
    fn batch() -> Batch {
        Batch {
            contract: BATCH_CONTRACT.to_string(),
            sample_dims: vec!["sample".to_string()],
            sample_shape: vec![1],
            n_time: 4,
            times: vec![0.0, 1.0, 2.0, 3.0],
            values: vec![8.0, 4.0, 2.0, 1.0],
            sd: None,
            se: None,
            n: None,
            nominal_time: None,
            lloq: None,
            max_doses: 1,
            dose_amount: vec![10.0],
            dose_time: vec![0.0],
            dose_duration: vec![f64::NAN],
            time_unit: "hr".to_string(),
            unit: "mg/l".to_string(),
            dose_unit: Some("mg".to_string()),
            substance: Some("caffeine".to_string()),
            substances: None,
            route: Some(Route::Oral),
            routes: None,
            tissue: None,
            coords: Vec::new(),
        }
    }

    /// The request of a batch and the options to analyse it with, as the wire carries it.
    fn request_of(batch: Batch, options: NcaOptions) -> Vec<u8> {
        let request = AnalysisRequest {
            contract: ANALYSIS_CONTRACT.to_string(),
            batch,
            options,
        };
        serde_json::to_vec(&request).unwrap()
    }

    /// The values of one column of an answer, by name.
    fn column(out: &Aligned, name: &str) -> Vec<f64> {
        let header: Value = serde_json::from_str(&header_text(out)).unwrap();
        let n_rows = header["n_rows"].as_u64().unwrap() as usize;
        let values = payload(out);
        let mut at = 0;
        for column in header["columns"].as_array().unwrap() {
            let width = column["width"].as_u64().unwrap() as usize;
            let take = width * n_rows;
            if column["name"] == name {
                return values[at..at + take].to_vec();
            }
            at += take;
        }
        panic!("no column {name} in {}", header["columns"]);
    }

    #[test]
    fn a_single_dose_request_answers_the_columns_and_the_values() {
        let out = analyze(&request_of(batch(), NcaOptions::default())).unwrap();
        let header: Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["contract"], ANALYSIS_CONTRACT);
        assert_eq!(header["n_rows"], 1);
        assert_eq!(header["sample_dims"], json!(["sample"]));
        let names: Vec<&str> = header["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|column| column["name"].as_str().unwrap())
            .collect();
        assert!(names.contains(&"auc_last"), "{names:?}");
        assert!(names.contains(&"cmax") && names.contains(&"flags"));
        // every column says what a row which does not report it contributes, which is what a
        // host merging the shards of one batch fills with
        let described = header["columns"].as_array().unwrap();
        let fill = |name: &str| {
            described
                .iter()
                .find(|column| column["name"] == name)
                .unwrap()["fill"]
                .clone()
        };
        assert_eq!(fill("cmax"), json!("NaN"));
        assert_eq!(fill("flags"), json!(0.0));
        assert_eq!(fill("accepted"), json!(0.0));
        // `c0_method` is a column an extravascular row does not report at all, and the one
        // whose fill is a code rather than a missing number
        assert_eq!(fill_of("c0_method"), json!(0.0));
        assert_eq!(fill_of("cl"), json!("NaN"));
        assert_eq!(column(&out, "cmax"), vec![8.0]);
        assert_eq!(column(&out, "tmax"), vec![0.0]);
        // the peak of this oral curve is its first point, which is `NcaFlag::NO_ABSORPTION`
        assert_eq!(column(&out, "flags"), vec![16.0]);
        // the half-life of a curve which halves every hour, and the unit of the column
        let thalf = column(&out, "thalf")[0];
        assert!((thalf - 1.0).abs() < 1e-12, "{thalf}");
        let unit = header["columns"]
            .as_array()
            .unwrap()
            .iter()
            .find(|column| column["name"] == "auc_last")
            .unwrap()["unit"]
            .as_str()
            .unwrap()
            .to_string();
        assert_eq!(unit, "hour * milligram / liter");
    }

    #[test]
    fn the_payload_is_as_long_as_the_columns_declare() {
        let out = analyze(&request_of(batch(), NcaOptions::default())).unwrap();
        let header: Value = serde_json::from_str(&header_text(&out)).unwrap();
        let n_rows = header["n_rows"].as_u64().unwrap() as usize;
        let declared: usize = header["columns"]
            .as_array()
            .unwrap()
            .iter()
            .map(|column| column["width"].as_u64().unwrap() as usize * n_rows)
            .sum();
        assert_eq!(payload(&out).len(), declared);
    }

    #[test]
    fn a_request_which_is_no_json_is_malformed_and_names_the_position() {
        let out = analyze(br#"{"contract":"#).unwrap();
        let header: Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["error"]["kind"], "malformed");
        let message = header["error"]["message"].as_str().unwrap();
        assert!(message.contains("line 1"), "{message}");
    }

    #[test]
    fn a_request_which_is_no_utf8_is_malformed() {
        let out = analyze(&[0x7b, 0xff]).unwrap();
        assert_eq!(kind_text(&out), "malformed");
    }

    #[test]
    fn another_contract_is_malformed_and_a_refusal_is_refused() {
        let mut skewed: Value =
            serde_json::from_slice(&request_of(batch(), NcaOptions::default())).unwrap();
        skewed["contract"] = json!("analysis/2");
        let out = analyze(skewed.to_string().as_bytes()).unwrap();
        assert_eq!(kind_text(&out), "malformed");

        // an effect curve is an analysis this build does not carry
        let effect = NcaOptions {
            kind: Kind::Effect,
            ..NcaOptions::default()
        };
        let out = analyze(&request_of(batch(), effect)).unwrap();
        assert_eq!(kind_text(&out), "refused");
        let header: Value = serde_json::from_str(&header_text(&out)).unwrap();
        let message = header["error"]["message"].as_str().unwrap();
        assert!(message.contains("Kind::Effect"), "{message}");
    }

    #[test]
    fn a_non_finite_value_travels_in_both_directions() {
        // the wire carries a NaN as the string "NaN", which is the encoding of
        // `pkpdutils_abi::nonfinite`, and a missing point is not measurable: the curve ends at
        // the third
        let mut wire: Value =
            serde_json::from_slice(&request_of(batch(), NcaOptions::default())).unwrap();
        wire["batch"]["values"] = json!([8.0, 4.0, 2.0, "NaN"]);
        let out = analyze(wire.to_string().as_bytes()).unwrap();
        assert_eq!(kind_text(&out), "");
        assert!(column(&out, "auc_last")[0].is_finite());
        assert_eq!(column(&out, "tlast"), vec![2.0]);

        // and a parameter which does not exist is a NaN of the payload, not a zero and not a
        // hole: a curve of nothing but missing points has no peak
        wire["batch"]["values"] = json!(["NaN", "NaN", "NaN", "NaN"]);
        let out = analyze(wire.to_string().as_bytes()).unwrap();
        assert!(column(&out, "cmax")[0].is_nan());
        assert!(column(&out, "auc_last")[0].is_nan());
        // `NcaFlag::NO_DATA`, which is what a page reads the row by
        assert_eq!(column(&out, "flags"), vec![64.0]);
    }

    #[test]
    fn a_panic_leaves_a_panic_output_behind() {
        crate::output::install_panic_hook();
        let caught = std::panic::catch_unwind(|| panic!("a bug of this build"));
        assert!(caught.is_err());
        let text = crate::output::published_text().expect("the hook published an output");
        let header: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(header["error"]["kind"], "panic");
        let message = header["error"]["message"].as_str().unwrap();
        assert!(message.contains("a bug of this build"), "{message}");
        crate::output::publish(None);
    }
}
