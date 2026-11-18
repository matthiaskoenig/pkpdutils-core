//! The output buffer of this host and the layout a page reads it with.
//!
//! One call writes one output, which stays alive until the next call writes the next one, and
//! `pkpd_output_ptr` and `pkpd_output_len` are how a page finds it. The layout is the one of
//! the engine of `sbmlsim-core`, so a page which drives both modules learns it once:
//!
//! ```text
//! | u32 header_len | header (JSON, header_len bytes) | zero padding | f64 values |
//! ```
//!
//! The padding is what makes the values start at a multiple of 8 counted from the start of the
//! output, and the output itself is allocated 8 aligned ([`Aligned`]), so the offset of the
//! values in the memory of the instance is a multiple of 8 as well and a page reads them in
//! place with a `Float64Array` rather than copying them. The numbers are little endian, which
//! is what wasm is, and row major: one column after the other, every column one value per
//! sample, in the order the header declares.
//!
//! A success header describes the answer and carries no numbers; an error header is
//! `{"error":{"kind":"...","message":"..."}}` and is followed by no values. The four kinds are
//! `refused`, `malformed`, `internal` (the three classes of [`ErrorClass`]) and `panic`, which
//! the hook of [`crate::analyze`] writes before the trap.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;

use pkpdutils_core::{ErrorClass, KernelError};

/// The largest output this host builds, 2 GiB.
///
/// A result of 51 columns over 100 000 samples is 40 MB and well inside it. The ceiling is here
/// because the allocator of a wasm instance aborts the instance when it cannot grow, which is a
/// trap a page cannot read: a request whose answer would be absurdly large is answered with
/// [`ErrorClass::Internal`] instead, while an allocation which fails below the ceiling is the
/// trap it is, reported by the loader as the memory it is.
pub(crate) const MAX_OUTPUT_BYTES: usize = 1 << 31;

/// The alignment of an output, so the values of a payload are 8 aligned in the memory.
const ALIGN: usize = 8;

/// A byte buffer aligned to [`ALIGN`], owned and freed on drop.
///
/// `Vec<u8>` is aligned to 1 and the allocator is free to hand out an odd address for it, which
/// would make the offset of the payload in the memory of the instance odd as well, however much
/// padding the header carries. This is the one allocation of the host which has to be aligned,
/// so it is the one which says so.
pub(crate) struct Aligned {
    ptr: *mut u8,
    len: usize,
}

impl Aligned {
    /// A zeroed buffer of `len` bytes, `None` when the allocation failed or `len` is 0.
    fn new(len: usize) -> Option<Self> {
        if len == 0 {
            return None;
        }
        let layout = Layout::from_size_align(len, ALIGN).ok()?;
        // SAFETY: the layout is non-zero and its alignment is a power of two.
        let ptr = unsafe { alloc_zeroed(layout) };
        if ptr.is_null() {
            return None;
        }
        Some(Self { ptr, len })
    }

    /// The bytes.
    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` is the live allocation of `len` bytes made in `new`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// The bytes, to write into.
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and `&mut self` is the only borrow.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// The address of the first byte, as a page reads it.
    pub(crate) fn addr(&self) -> usize {
        self.ptr as usize
    }
}

impl Drop for Aligned {
    fn drop(&mut self) {
        if let Ok(layout) = Layout::from_size_align(self.len, ALIGN) {
            // SAFETY: the pointer and the layout are the ones of the allocation of `new`.
            unsafe { dealloc(self.ptr, layout) };
        }
    }
}

thread_local! {
    /// The output of the last call, read by `pkpd_output_ptr` and `pkpd_output_len`.
    static OUTPUT: RefCell<Option<Aligned>> = const { RefCell::new(None) };
}

/// Hand an output to the page and drop the one before it.
pub(crate) fn publish(out: Option<Aligned>) {
    OUTPUT.with_borrow_mut(|slot| *slot = out);
}

/// The offset of the payload for a header of `header_len` bytes.
fn payload_start(header_len: usize) -> usize {
    (4 + header_len).next_multiple_of(ALIGN)
}

/// One success output: the header and the columns of the payload in order.
///
/// Every part is one column, or several columns of one variable laid out one after the other,
/// and they are written in the order they are given, which is the order the header declares.
pub(crate) fn ok_from(header: &str, parts: &[&[f64]]) -> Option<Aligned> {
    let values: usize = parts.iter().map(|part| part.len()).sum();
    let start = payload_start(header.len());
    let len = start.checked_add(values.checked_mul(8)?)?;
    if len > MAX_OUTPUT_BYTES {
        return None;
    }
    let mut out = Aligned::new(len)?;
    let bytes = out.as_mut_slice();
    bytes[..4].copy_from_slice(&u32::try_from(header.len()).ok()?.to_le_bytes());
    bytes[4..4 + header.len()].copy_from_slice(header.as_bytes());
    let mut at = start;
    for part in parts {
        for value in *part {
            bytes[at..at + 8].copy_from_slice(&value.to_le_bytes());
            at += 8;
        }
    }
    Some(out)
}

/// One error output: a header of the kind and the message, and no payload.
///
/// The message is escaped by `serde_json`, so a sentence of the kernel which carries a quote or
/// a newline arrives as that sentence and not as a broken header.
pub(crate) fn error(kind: &str, message: &str) -> Option<Aligned> {
    let header = serde_json::json!({"error": {"kind": kind, "message": message}});
    ok_from(&header.to_string(), &[])
}

/// The kind an error of the kernel is reported as, from its class.
///
/// The three classes answer three different things and a page routes on them: `refused` is an
/// analysis this build does not carry, `malformed` is a request or a contract to fix, and
/// `internal` is a bug of this build.
pub(crate) fn kind_of(error: &KernelError) -> &'static str {
    match error.class() {
        ErrorClass::Refused => "refused",
        ErrorClass::Malformed => "malformed",
        ErrorClass::Internal => "internal",
    }
}

/// The output of an answer which would be larger than [`MAX_OUTPUT_BYTES`].
pub(crate) fn too_large(bytes: usize) -> Option<Aligned> {
    error(
        "internal",
        &format!(
            "the answer is {bytes} bytes, more than this host writes in one output \
             ({MAX_OUTPUT_BYTES}); analyse the batch in shards"
        ),
    )
}

/// The bytes of a request a page wrote into an allocation of [`pkpd_alloc`].
///
/// # Safety
///
/// `ptr` is an address `pkpd_alloc` returned and `len` the number of bytes written into it. The
/// slice borrows the memory of the page for as long as the caller holds it, which is the one
/// call it belongs to: an entry point reads the request and does not keep it.
pub(crate) unsafe fn read_request<'a>(ptr: usize, len: usize) -> &'a [u8] {
    if ptr == 0 || len == 0 {
        return &[];
    }
    // SAFETY: the caller guarantees the range is the request it wrote.
    unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }
}

/// Whether an output is an error header, which is the return code of every entry point.
///
/// An error header is the one object whose first key is `error`, which [`error`] builds and
/// nothing else does: a success header starts with the contract. Reading the prefix rather than
/// parsing the header again is what keeps a return code from costing a second deserialization.
/// An output which could not be allocated at all is an error as well, and the page sees it as
/// a length of 0.
pub(crate) fn is_error(out: Option<&Aligned>) -> bool {
    let Some(out) = out else {
        return true;
    };
    out.as_slice().get(4..4 + ERROR_PREFIX.len()) == Some(ERROR_PREFIX)
}

/// The first bytes of an error header.
const ERROR_PREFIX: &[u8] = br#"{"error":"#;

/// Install the hook which leaves a `panic` output behind, once per instance.
///
/// `wasm32-unknown-unknown` unwinds nothing: a panic of the kernel runs the hook and then traps,
/// and the page catches a `RuntimeError` with no message of its own. The hook is what turns that
/// into a sentence: it writes the panic as an error output of kind `panic`, which the loader
/// reads after it caught the trap. The hook before it is kept and called, so a panic of a test
/// of this crate still prints what it would have printed.
pub(crate) fn install_panic_hook() {
    use std::sync::Once;
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            publish(error("panic", &info.to_string()));
            previous(info);
        }));
    });
}

/// The header of the published output, for the tests of the panic hook.
#[cfg(test)]
pub(crate) fn published_text() -> Option<String> {
    OUTPUT.with_borrow(|slot| slot.as_ref().map(tests::header_text))
}

/// The address of the output of the last call, 0 when there is none.
///
/// A page reads `pkpd_output_len` bytes from here. The output lives until the next call, so a
/// page which wants to keep it past that copies it.
///
/// The addresses and the lengths of this ABI are `usize`, which is `u32` on `wasm32` and
/// therefore the `i32` of the wasm type system, so the exported signature is the one a page
/// expects. Written that way they are also the addresses of the host this crate is tested on,
/// where a pointer is 64 bits wide: an `i32` in the signature would make every one of these
/// four functions untestable outside wasm.
#[unsafe(no_mangle)]
pub extern "C" fn pkpd_output_ptr() -> usize {
    OUTPUT.with_borrow(|slot| slot.as_ref().map_or(0, Aligned::addr))
}

/// The length of the output of the last call in bytes, 0 when there is none.
///
/// A length of 0 after a call means the host could not allocate its own answer, which is a page
/// out of memory and the one case it cannot be told anything else about.
#[unsafe(no_mangle)]
pub extern "C" fn pkpd_output_len() -> usize {
    OUTPUT.with_borrow(|slot| slot.as_ref().map_or(0, |out| out.as_slice().len()))
}

/// `len` bytes for a page to write a request into, 8 aligned, 0 when it cannot be had.
///
/// A page allocates, writes its request, calls an entry point and frees. A `len` of 0 is no
/// allocation and answers 0, which a page is free to pass to `pkpd_free`.
#[unsafe(no_mangle)]
pub extern "C" fn pkpd_alloc(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let Ok(layout) = Layout::from_size_align(len, ALIGN) else {
        return 0;
    };
    // SAFETY: the layout is non-zero and its alignment is a power of two.
    let ptr = unsafe { alloc_zeroed(layout) };
    ptr as usize
}

/// Give back what `pkpd_alloc` handed out.
///
/// # Safety
///
/// `ptr` is an address `pkpd_alloc` returned and `len` the length it was asked for. A null
/// pointer and a zero length are a no-op, so a page may free an allocation it never got.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pkpd_free(ptr: usize, len: usize) {
    if ptr == 0 || len == 0 {
        return;
    }
    let Ok(layout) = Layout::from_size_align(len, ALIGN) else {
        return;
    };
    // SAFETY: the caller guarantees the pointer and the length are the ones of an allocation of
    // `pkpd_alloc`, which used this very layout.
    unsafe { dealloc(ptr as *mut u8, layout) };
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// The header of an output, as a string.
    pub(crate) fn header_text(out: &Aligned) -> String {
        let bytes = out.as_slice();
        let len = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        String::from_utf8(bytes[4..4 + len].to_vec()).unwrap()
    }

    /// The kind of an error output, the empty string for an answer.
    pub(crate) fn kind_text(out: &Aligned) -> String {
        let header: serde_json::Value = serde_json::from_str(&header_text(out)).unwrap();
        header["error"]["kind"]
            .as_str()
            .unwrap_or_default()
            .to_string()
    }

    /// The values of the payload of an output.
    pub(crate) fn payload(out: &Aligned) -> Vec<f64> {
        let bytes = out.as_slice();
        let len = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        bytes[payload_start(len)..]
            .chunks(8)
            .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn the_payload_starts_at_a_multiple_of_eight() {
        let header = r#"{"columns":["auc_last"],"n_rows":3}"#;
        let out = ok_from(header, &[&[1.0, 2.0, 3.0]]).unwrap();
        let bytes = out.as_slice();
        let header_len = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
        assert_eq!(header_len, header.len());
        assert_eq!(&bytes[4..4 + header_len], header.as_bytes());
        let start = 4 + header_len;
        let padded = payload_start(header_len);
        assert_eq!(padded % 8, 0);
        assert!(bytes[start..padded].iter().all(|byte| *byte == 0));
        assert_eq!(bytes.len(), padded + 24);
        assert_eq!(payload(&out), vec![1.0, 2.0, 3.0]);
        // and the allocation itself is aligned, so the offset in the memory of an instance is
        // a multiple of 8 as well
        assert_eq!(out.addr() % 8, 0);
    }

    #[test]
    fn the_parts_of_a_payload_are_written_in_order() {
        let out = ok_from(r#"{"n_rows":2}"#, &[&[1.0, 2.0], &[], &[3.0]]).unwrap();
        assert_eq!(payload(&out), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn a_nan_of_the_payload_keeps_its_bits() {
        let out = ok_from(r#"{"n_rows":1}"#, &[&[f64::NAN, f64::INFINITY, -0.0]]).unwrap();
        let values = payload(&out);
        assert!(values[0].is_nan());
        assert_eq!(values[1], f64::INFINITY);
        assert_eq!(values[2].to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn an_error_output_carries_the_kind_of_the_class() {
        let refused = KernelError::Unsupported {
            what: "the analysis of the moon".to_string(),
        };
        let out = error(kind_of(&refused), &refused.to_string()).unwrap();
        let header: serde_json::Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["error"]["kind"], "refused");
        assert!(
            header["error"]["message"]
                .as_str()
                .unwrap()
                .contains("the analysis of the moon")
        );
        assert!(payload(&out).is_empty());
        assert_eq!(kind_of(&KernelError::Batch("bad".to_string())), "malformed");
        assert_eq!(
            kind_of(&KernelError::Options("bad".to_string())),
            "malformed"
        );
        assert_eq!(
            kind_of(&KernelError::Internal("a bug".to_string())),
            "internal"
        );
    }

    #[test]
    fn a_message_with_a_quote_or_a_newline_is_escaped() {
        let out = error("malformed", "a \"quoted\" word\nand a line").unwrap();
        let header: serde_json::Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["error"]["message"], "a \"quoted\" word\nand a line");
    }

    #[test]
    fn an_answer_past_the_ceiling_is_an_internal_error_and_no_allocation() {
        // 2 GiB of f64 is 2^28 values, and the host does not try to allocate the answer
        let huge = MAX_OUTPUT_BYTES / 8;
        assert!(ok_from(r#"{"n_rows":1}"#, &[&[0.0; 0]]).is_some());
        let out = too_large(huge * 8).unwrap();
        let header: serde_json::Value = serde_json::from_str(&header_text(&out)).unwrap();
        assert_eq!(header["error"]["kind"], "internal");
        let message = header["error"]["message"].as_str().unwrap();
        assert!(message.contains(&(huge * 8).to_string()), "{message}");
        assert!(message.contains("shards"), "{message}");
    }

    #[test]
    fn an_error_output_is_told_from_an_answer_by_its_prefix() {
        let answer = ok_from(r#"{"contract":"analysis/1","n_rows":1}"#, &[&[1.0]]).unwrap();
        assert!(!is_error(Some(&answer)));
        let failed = error("malformed", "no").unwrap();
        assert!(is_error(Some(&failed)));
        // an output which could not be allocated is an error the page sees as a length of 0
        assert!(is_error(None));
    }

    #[test]
    fn the_output_is_empty_before_the_first_call() {
        publish(None);
        assert_eq!(pkpd_output_len(), 0);
        assert_eq!(pkpd_output_ptr(), 0);
    }

    #[test]
    fn a_published_output_is_the_one_a_page_reads() {
        publish(ok_from(r#"{"n_rows":1}"#, &[&[7.0]]));
        let len = pkpd_output_len();
        let ptr = pkpd_output_ptr();
        assert!(ptr != 0 && len == payload_start(12) + 8);
        // SAFETY: the address and the length are the ones of the live output.
        let bytes = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) };
        let value = f64::from_le_bytes(bytes[payload_start(12)..].try_into().unwrap());
        assert_eq!(value, 7.0);
        publish(None);
    }

    #[test]
    fn a_zero_length_allocation_is_a_null_pointer_and_a_free_of_it_is_harmless() {
        assert_eq!(pkpd_alloc(0), 0);
        // the length a page passes when it sends -1 as an i32, which is no allocation either
        assert_eq!(pkpd_alloc(usize::MAX), 0);
        // SAFETY: a null pointer and a zero length are documented as a no-op.
        unsafe { pkpd_free(0, 0) };
        // SAFETY: as above, with a length a page might pass next to the null pointer.
        unsafe { pkpd_free(0, 16) };
    }

    #[test]
    fn an_allocation_is_aligned_and_round_trips() {
        let ptr = pkpd_alloc(24);
        assert!(ptr != 0);
        assert_eq!(ptr % 8, 0);
        // SAFETY: the allocation is live and 24 bytes long.
        unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, 24) }.fill(1);
        // SAFETY: the pointer and the length are the ones of the allocation above.
        unsafe { pkpd_free(ptr, 24) };
    }
}
