//! The python binding of the pkpdutils kernel.
//!
//! This is the only crate of the workspace which knows that python exists. It takes what
//! `pkpdutils` already holds in memory - the numpy buffers of a `Timecourses` batch and the
//! fields of an `NCAOptions` - turns them into the wire shapes of [`pkpdutils_abi`] and hands
//! them to the kernel. Nothing travels as JSON: the layouts of the two sides already agree, so
//! a conversion here is a copy of a buffer and not a parse.
//!
//! The crate binds pyo3 against the stable ABI from python 3.13 on (`abi3-py313`), which is the
//! floor of `pkpdutils` itself, so that one wheel per platform serves every python the package
//! supports. It cannot be built for `wasm32`, which is why continuous integration excludes it
//! from the wasm gates by name; the kernel it binds is checked there as before.

mod convert;
