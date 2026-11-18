//! What this build is, which a page reads once before it sends anything.

use serde_json::json;

use crate::output::{Aligned, answer, publish};

/// The version of the output layout of this host.
///
/// A page which reads another number than this one is talking to a module it does not know how
/// to read, which is a deployment to fix rather than something to work around: the loader of
/// the npm package refuses such a module instead of guessing at its answers.
pub(crate) const OUTPUT_FORMAT: u32 = 1;

/// The entry points this build carries, in the order they were added.
///
/// A page reads this list rather than probing for a symbol. `time_to_steady_state`,
/// `nca_sparse`, `nca_urine` and the draws of the bootstrap are in the kernel and not exported
/// here yet; each is the shape of `pkpd_summarize` over a request type which already exists, so
/// adding one is a longer list here and no change of any contract.
const ENTRY_POINTS: [&str; 3] = ["analyze", "summarize", "about"];

/// What this build is, the pure function the export goes through.
pub(crate) fn about() -> Option<Aligned> {
    let header = json!({
        "version": env!("CARGO_PKG_VERSION"),
        "output_format": OUTPUT_FORMAT,
        "contracts": {
            "analysis": pkpdutils_abi::ANALYSIS_CONTRACT,
            "batch": pkpdutils_abi::BATCH_CONTRACT,
        },
        "entry_points": ENTRY_POINTS,
    });
    answer(&header.to_string(), &[])
}

/// What this build is: the version, the output layout, the contracts and the entry points.
///
/// Returns 0, since there is nothing about this answer which can fail that would not have
/// failed the instantiation of the module.
#[unsafe(no_mangle)]
pub extern "C" fn pkpd_about() -> i32 {
    crate::output::install_panic_hook();
    let out = about();
    let failed = i32::from(crate::output::is_error(out.as_ref()));
    publish(out);
    failed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::tests::{header_text, payload};

    #[test]
    fn about_names_this_version_the_layout_and_the_contracts() {
        let out = about().unwrap();
        let header: serde_json::Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(header["output_format"], OUTPUT_FORMAT);
        assert_eq!(
            header["contracts"]["analysis"],
            pkpdutils_abi::ANALYSIS_CONTRACT
        );
        assert_eq!(header["contracts"]["batch"], pkpdutils_abi::BATCH_CONTRACT);
        assert_eq!(
            header["entry_points"],
            json!(["analyze", "summarize", "about"])
        );
        assert!(payload(&out).is_empty());
        assert_eq!(pkpd_about(), 0);
    }
}
