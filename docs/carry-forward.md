# Carry forward from the foundation plan

What the first plan of `pkpdutils-core` left open, deliberately. Every item here was seen, judged and not acted on, with the reason. The plan's own scratch workspace is not part of this repository and does not survive, so this file is the record.

The foundation plan built the workspace, the dependency catalogue, the wire enumerations and options, the terminal window types, the batch and result contracts, the unit registry with its pint equality gate, the parameter unit expressions and the vendored conformance oracles. It deliberately built no numerics: the NCA kernels, the hosts and the release are later plans.

## Required inputs for the later plans

**The terminal kernel must implement `tie_tolerance`.** The Task 1 spike selected the terminal window with a strict greater-than, while `nca/terminal.py` applies `tie_tolerance` (1e-4 by default), which prefers the window with more points among near-best candidates. A strict comparison is a different selection rule and diverges from the validation oracles on noisy profiles. The spike is throwaway, so this is recorded here rather than in its code.

**`cdisc.py` holds 83 unit literals the pint gate cannot see.** `PKUNIT_SYMBOLS` and `_PKUNIT_SPELLINGS` in `src/pkpdutils/cdisc.py` feed every one of them to `parse_unit` at `import pkpdutils.cdisc` time (`cdisc.py:261`), against the 44 literals `scripts/pint_gate.py` extracts today. Under the adapter they would fail at import rather than as a test failure. All 83 were probed against both engines during the final review with zero disagreements in long and short form, so nothing is broken now; the plan that writes the adapter owns extending the extraction to the data tables.

**Package metadata blocks publishing.** The workspace manifests carry no `description`, `repository`, `readme` or `rust-version`, and `cargo doc` is not in CI. Publishing is not in this plan, so this belongs to the plan that releases.

**Serializing a batch clones it.** `BatchWire::from(self.clone())` at `crates/abi/src/batch.rs:575` copies every array, and a request clones it again, so peak memory doubles at 20 000 rows. A borrowed wire struct removes it. Left to the plan that owns the hosts and the performance gates, where it can be measured instead of guessed.

**A batch with zero time points is accepted.** `n_time` (`crates/abi/src/batch.rs:208`) is unconstrained, and `an_empty_optional_array_is_not_an_absent_one` (`batch.rs:737`) now relies on that, since it is how an empty optional array is distinguished from an absent one. Whether a real batch may have no time points is a question for the plan that builds one.

## Mirrors that must not be tightened here

`check_dose_unit` in `src/pkpdutils/units.py` accepts `dimensionless`, `none`, `count`, `%`, `mmol/mmol`, `IU/IU` and `mg/g` as dose units, and `is_per_bodyweight` returns `True` for every one of them. The Rust registry mirrors this deliberately, with a doc comment and a test pinning it. The binding goal of this crate is agreement with Python: a unit Rust rejects which Python accepts turns a batch that analyses today into a boundary failure tomorrow. It is almost certainly wrong on its own terms, since a dose of "percent" is not a dose, but it is fixed in `pkpdutils` first and in the registry second, never the other way round.

Three wire shapes depart from the "a variant enum is externally tagged" constraint because Python spells them otherwise: `Blq` and `BlqSetting` untagged, `LabelWire` adjacently tagged. All three are forced by `BLQAction | float`, `BLQHandling | BLQRules` and `{"type", "value"}`.

`fm` resolves to femtometre rather than to pint's femtomolar. It is the one spelling of the registry that names a different unit from pint's, recorded in the `pkpdutils_units` crate documentation, and no caller can observe it because femtomolar reaches no PK parameter.

## Open minors

None of these is a correctness defect. They are listed so the next reader does not rediscover them.

- `NcaOptions::validate` reaches `TerminalPhase::validate` twice, at `options.rs:720` and `options.rs:867`, so the same object is validated three times per deserialize.
- `"n_points must be at least 3"` (`options.rs:455`) and `"n_points_min must be at least 2"` (`options.rs:288`) do not name the value found, unlike every message routed through `check`/`check_opt`.
- A standalone `TerminalPhase` serializes its windows as the record list, a shape Python's `TerminalPhase.model_dump` never produces. Harmless while only `NcaOptions` crosses the boundary; it wants a doc line on the field before a bare `TerminalPhase` is sent to Python.
- The untagged `Blq` and `BlqSetting` report "data did not match any variant of untagged enum Blq" instead of the worded message of `BlqRules::validate`, and a misspelled key inside `blq` is rejected without being named. A hand-written `Deserialize` branching on string against map would forward the inner error. Documented, not fixed: real complexity for an error surface.
- `SampleCoordinate` (`batch.rs:97`) carries no unit, where `NCAResult.dose_normalized` reads one from the variable's attributes (`nca/result.py:244`). Derivable from `batch.dose_unit`, fails loudly rather than silently, and dose normalization has no Rust counterpart yet.
- `check_coords` (`batch.rs:108`) checks duplicate names in O(n^2). An error path, and coordinate counts are single digits.
- `Attributes::of` (`result.rs:191`) writes `substance: null` where Python omits the key. The adapter reads a typed field, so absent and null are the same thing to it.
- `assert_ne!(batch.route, infusion.route)` (`batch.rs:778`) is weak: row 0 shares the batch route, so only row 1 proves the row route wins.
- The empty placeholder `{}` and a bare closing brace in a parameter unit expression have no test. Both error today, `{}` as an `UnknownPlaceholder` with an empty name.
- `crates/abi/src/options.rs` is 911 lines and `crates/units/src/registry.rs` 837, each one responsibility. `options/{blq,terminal,nca}.rs` and splitting the registry's tokenizer into `parse.rs` would make each unit independently readable; both reviewers judged them coherent as they stand.
- The `impl ... ALL` blocks added for `C0Method` and `Route` sit beside the pre-existing `CODE_*`/`is_iv` blocks instead of being folded into them.
- On a failed restore, `scripts/vendor_oracles.py` leaves the old vendoring in `.oracles-old` and surfaces its own error with the original as `__context__`. A double failure, and `.gitignore` already covers the leftover.
- The source-side reads of `scripts/vendor_oracles.py`, `sha256(source / rel)` and `copyfile`, are unguarded, and `is_file` follows a tracked symlink. The paths come from `git ls-files` at a commit on a tree required to be clean, git cannot track a FIFO, and a tracked symlink pointing at one would block a human-run command rather than a gate.
- A symlinked `MANIFEST.json` fails as "not a regular file" rather than with the symlink wording of the checksum loop, and a missing one reports the same. Both exit non-zero and name the file.
- `rel.to_str().unwrap()` in the oracle test panics on a non-UTF-8 name. A test failure rather than a hang, and the vendored set comes from `git ls-files` over ASCII paths.

## Method notes worth reusing

**Prove a guard with defeating input, not by inspection.** Everything load-bearing in this plan was verified by making it fail: a FIFO at an oracle path (which found a hang two rounds of enumeration had missed), `math.nan` injected into the gate generator, a monkeypatched `Path.rename` to exercise the restore path, a tampered checksum, and a zero-record fixture. Each found something reading the code had not.

**Watch for an assertion that closes a loop with itself.** Three separate defects in this plan had that shape: a gate whose record count was asserted against its own regenerated header, a manifest self-consistent with a vendoring that no longer existed, and a hand-written axis list that was the only thing deciding which units got gated. The fix each time was an independent literal or a derivation from a different source.

**A number in a report should be re-derivable from the repository.** A probe that is not committed cannot be cited, however right its conclusion.

**Remove generated paths with `rm -f "${DIR:?}"/"${NAME:?}"`.** An unset variable then fails the shell instead of expanding to a filesystem root.

## Documents behind the code

The design at `docs/superpowers/specs/2026-11-15-rust-core-design.md` in `pkpdutils` describes `batch/1` without `tissue`, and Task 5 Step 4 of the plan specifies six result fields. Both are behind this repository in exactly those two places: `Batch` carries `tissue`, and `AnalysisResult` carries `attributes` and the boolean and text sections. They are approved documents, so they were left for their author to amend.

# Carry forward from the single dose kernel

What the plan of the single dose non-compartmental analysis kernel left open, deliberately. The rule of the first section holds here as well: every item was seen, judged and not acted on. None of them is a wrong number. The final review of the branch found no critical defect and no reachable panic, and measured the kernel against the python package over 73 analyses and 3 409 numbers with every one of them agreeing, 93.3 percent of them bit for bit and the worst relative difference at 2.65e-15.

## The gates do not compose

**Nine of the 54 parameter columns are named by no gate at all** (the 53 of `PARAMETERS` in `crates/core/src/parameters.rs` plus `c0_method`): `cmin`, `tmin`, `aumc_all`, `thalf_eff`, `lambda_z_stderr`, `aumc_back_extrap_fraction`, `cmax_half`, `tmax_half` and `c0_method`, and fifteen more are reached by the benchmark gate alone.

**The whole result layer is ungated.** No gate reads `flags`, `accepted`, `excluded`, `n`, the `dose_amount` coordinate or the `candidate_*` point variables.

**Ungated input classes:** every BLQ action and axis and both `lloq` paths, `TerminalMethod::LastN`, `TerminalMethod::Manual` and the per-sample windows, `min_adj_r2`, any non-default `tie_tolerance`, `AucMethod::Log`, `C0Method::FirstValue` and `C0Method::None`, mixed-route batches, unsorted and duplicate times, and pre-dose samples.

The numbers of all of it were checked against python by the reviewer of the branch, so this is regression risk and not a defect: a later plan can break any of those paths and every gate still passes. The cheapest closure is one committed snapshot fixture of the kernel's own output over about a dozen curves which cover those classes, generated once from the python package.

## Five findings of Task 5 are lost

The ledger of Task 5 records "5 minors deferred" and neither it nor the task report enumerates them. They are unrecoverable, so no list of open items of this branch, this one included, can be assumed to be complete.

## Open items

- `window_statistics` (`crates/core/src/terminal.rs:101`) clamps `columns` to the shortest of its four slices, so a caller who passes a short `out` gets windows over a truncated row with silently wrong suffix sums; a `debug_assert_eq!` on the widths closes it.
- `crates/core/src/parameters.rs:936` allocates `vec![false; t.len()]` per row under `TerminalMethod::Manual`, the one allocation `Scratch` exists to avoid, and no benchmark shape uses `Manual`, so the cost of it is unmeasured.
- The backward summation order of the areas is pinned by no committed test; only the external cross-check against python covers it. A `to_bits` test of a long row against a python value would make it a committed requirement.
- The strictness of `slope < 0.0` is unpinned at the boundary: a constructed zero-slope window should yield `NcaFlag::POSITIVE_SLOPE` and no test says so.
- A batch of zero samples returns a result with a single column, because `parameter_columns` (`crates/core/src/assemble.rs:176`) derives the reported set from the routes of the rows and an empty batch has none, where python returns the full 37-variable set, which the reviewer of the branch measured; deriving the set from `batch.route`/`batch.routes` when there are no rows is the faithful fix.
- Python's `test_units_of_the_references` has no rust owner: the validation gate pins the dimensions by converting into the reference unit instead of comparing the unit string python asserts. The reviewer of the branch compared the strings over 73 analyses and all of them matched, so the behaviour is right and only the gate is missing.
- Lift `Table`, `field` and `number` into `crates/oracles/tests/common/csv.rs` behind a `#[path]` include: three gate binaries carry the reader today and the version in `benchmarks.rs` is the strict superset of the others.
- `read_text` of `crates/oracles/tests/common/mod.rs:27` ends with `fs::read_to_string(path).unwrap()`, which drops the path on a read error after `symlink_metadata` has succeeded.
- One documentation pass over `docs/python-reference/gates.md:7`: the citation `:15` of `crates/oracles/tests/common/mod.rs` is off by one, since the expression it quotes is on `:16`; "16 regression numbers" counts value-level assertions, 14 numeric and 2 NaN, where the surrounding text counts numbers; and `f_minute / f_hour` is called a "prefix ratio" in `crates/oracles/tests/common/units.rs:36` where it is 60.
- `Batch::n_doses_of_row` (`crates/abi/src/batch.rs:423`) asserts on the row and then slices, so it panics on an unvalidated batch even for row 0. It predates this branch, and the kernel's deliberate avoidance of it is the evidence that it is a trap for the next host.
- The repository pins no ruff configuration, so `ruff check` over `scripts/` means whatever the installed ruff selects by default: ruff 0.17.5 reports ten findings there, none of them in the kernel or in a fixture, six of them `B023` on closures which are called inside the iteration that defines them (`scripts/python_timing.py`, benign) and one of them an import grouping which contradicts the first-party convention of the `pkpdutils` repository. A committed `ruff.toml` which selects the rules and names `pkpdutils` first party, and then one pass over what it reports, is the fix.
- A half padded dose - an amount without a time or a time without an amount - is accepted where `Timecourses.from_arrays` raises ("A dose needs an amount and a time, or both NaN for the padding"). It is harmless after the padding check which `Batch::validate` now carries, because the dose columns of a row are then a prefix of it in each of the two arrays and the positional pick of `reference_dose` therefore lands on a column which is a dose or on no dose at all; the host plan can refuse it outright and name the column, as python does.
