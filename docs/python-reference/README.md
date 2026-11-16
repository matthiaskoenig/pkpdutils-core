# The Python kernel, written down

These files describe what `pkpdutils` computes, extracted from its source at the commit the oracles were vendored from, so that the Rust port has one written reference instead of a reading of the Python per task.

They are a description, not a design: they say what the Python does, including where it is odd, and they take no position on what the Rust should look like. Agreement with the Python is the binding goal of the port, so an oddity recorded here is mirrored rather than corrected, with a doc comment saying so.

| file | covers |
|---|---|
| `areas.md` | the pipeline order of `compute_parameters`, `pack_valid`, the segment formulas of all three AUC methods, every single-dose parameter with its formula and its NaN conditions, the value inserted at the dose time, BLQ, the predicted family, dose normalization |
| `terminal.md` | `window_statistics` and its exact summation order, the four terminal selection rules with the tie tolerance, the candidate table, the reported window quantities, every flag with the condition that sets it |
| `gates.md` | the three Python test gates, their case counts and tolerances, the structure of both reference files, and which of the vendored oracles each gate uses |

Each entry carries `file:line` references into the Python. They are accurate for `pkpdutils` at the commit in `testdata/oracles/MANIFEST.json` and go stale when that commit moves; the line numbers are the first thing to distrust.
