"""Vendor the conformance oracles of `pkpdutils` into `testdata/oracles/`.

Copies the reference files the Python tests pin their numbers to, keeping their
relative paths, and writes `MANIFEST.json` with the sha256 of every file and the
commit of the `pkpdutils` repository they were copied from. It refuses a source
tree with uncommitted changes, since the recorded commit would not describe the
copied content.

With `--check` nothing is copied. The vendored files are re-hashed against the
manifest and, when `--source` is given, the source files against it as well and
the commit of the source against the recorded one. It exits non-zero on any
difference and names the files, including a file under `testdata/oracles/` the
manifest does not list. The set of files comes from `git ls-files` of the source,
and a change of that set against an existing manifest needs `--accept-changes`.
Standard library only.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent.parent
ORACLE_DIR = REPO_DIR / "testdata" / "oracles"
MANIFEST_PATH = ORACLE_DIR / "MANIFEST.json"

FILES = [
    "tests/data/reference/nca_reference.json",
    "tests/data/validation/reference.json",
    "tests/data/validation/README.md",
]
DIRECTORIES = ["docs/data/benchmarks"]


def sha256(path: Path) -> str:
    """The sha256 of one file, hex encoded, as the manifest records it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest() -> dict:
    """Read MANIFEST.json, never opening a symlink or a FIFO (which would block)."""
    if MANIFEST_PATH.is_symlink() or not MANIFEST_PATH.is_file():
        sys.exit(f"refusing: {MANIFEST_PATH.name} is not a regular file")
    return json.loads(MANIFEST_PATH.read_text())


def git(source: Path, *args: str, strip: bool = True) -> str:
    """The standard output of one git command in a source tree."""
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip() if strip else result.stdout


def source_commit(source: Path) -> str:
    """Return the commit of a source tree without changes to tracked files."""
    dirty = git(source, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        sys.exit(
            f"refusing: {source} has uncommitted changes, the recorded commit "
            f"would not describe the copied files:\n{dirty}"
        )
    return git(source, "rev-parse", "HEAD")


def tracked(source: Path, *pathspecs: str) -> list[str]:
    """The paths the commit of the source tracks under some pathspecs."""
    out = git(source, "ls-files", "-z", "--", *pathspecs, strip=False)
    return [p for p in out.split("\0") if p]


def relative_paths(source: Path) -> list[str]:
    """The oracle set, the files the commit of the source tracks."""
    tracked_files = set(tracked(source, *FILES))
    absent = [p for p in FILES if p not in tracked_files]
    if absent:
        sys.exit(f"refusing: not tracked in the source commit: {', '.join(absent)}")
    paths = set(tracked_files)
    for directory in DIRECTORIES:
        found = tracked(source, directory)
        if not found:
            sys.exit(f"refusing: no tracked files under {directory} in the source")
        paths.update(found)
    missing = [p for p in sorted(paths) if not (source / p).is_file()]
    if missing:
        sys.exit(f"missing in source: {', '.join(missing)}")
    return sorted(paths)


def vendored_paths() -> set[str]:
    """Every path under the oracle directory, relative to it, directories left out."""
    return {
        p.relative_to(ORACLE_DIR).as_posix()
        for p in ORACLE_DIR.rglob("*")
        # everything but a real directory is a path: a symlink, a FIFO, a file
        if p.is_symlink() or not p.is_dir()
    }


def vendor(source: Path, accept_changes: bool) -> None:
    """Copy the oracle set of the source and write the manifest.

    The files go through a staging directory, so that a failure leaves the
    oracles which are vendored today in place.
    """
    commit = source_commit(source)
    paths = relative_paths(source)
    if MANIFEST_PATH.exists() or MANIFEST_PATH.is_symlink():
        old = set(read_manifest()["files"])
        added, removed = sorted(set(paths) - old), sorted(old - set(paths))
        if (added or removed) and not accept_changes:
            lines = [f"  added: {p}" for p in added] + [
                f"  removed: {p}" for p in removed
            ]
            sys.exit(
                "refusing: the oracle set changed, pass --accept-changes if that "
                "is intended:\n" + "\n".join(lines)
            )
    staging = ORACLE_DIR.parent / ".oracles-staging"
    backup = ORACLE_DIR.parent / ".oracles-old"
    for leftover in (staging, backup):
        shutil.rmtree(leftover, ignore_errors=True)
    try:
        files = {}
        for rel in paths:
            target = staging / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / rel, target)
            files[rel] = sha256(source / rel)
        manifest = {"source_commit": commit, "files": files}
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if ORACLE_DIR.exists():
            ORACLE_DIR.rename(backup)
        try:
            staging.rename(ORACLE_DIR)
        except BaseException:
            if backup.exists() and not ORACLE_DIR.exists():
                backup.rename(ORACLE_DIR)
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    shutil.rmtree(backup, ignore_errors=True)
    print(f"vendored {len(files)} files from pkpdutils {commit}")


def check(source: Path | None) -> int:
    """Re-hash the vendored files, and the source as well when it is given.

    Returns:
        The exit code, 1 on any difference.
    """
    manifest = read_manifest()
    recorded = manifest["files"]
    problems = []
    for rel, digest in recorded.items():
        path = ORACLE_DIR / rel
        if path.is_symlink():
            problems.append(f"vendored path is a symlink: {rel}")
        elif not path.exists():
            problems.append(f"vendored file missing: {rel}")
        elif not path.is_file():
            problems.append(f"vendored path is not a regular file: {rel}")
        elif sha256(path) != digest:
            problems.append(f"vendored file differs from manifest: {rel}")
    for rel in sorted(vendored_paths() - set(recorded) - {"MANIFEST.json"}):
        problems.append(f"vendored file not in manifest: {rel}")
    if source is not None:
        for rel in relative_paths(source):
            if rel not in recorded:
                problems.append(f"in source, not in manifest: {rel}")
            elif sha256(source / rel) != recorded[rel]:
                problems.append(f"source differs from manifest: {rel}")
        for rel in recorded:
            if not (source / rel).is_file():
                problems.append(f"in manifest, not in source: {rel}")
        commit = git(source, "rev-parse", "HEAD")
        if commit != manifest["source_commit"]:
            problems.append(
                f"source commit {commit} differs from recorded {manifest['source_commit']}"
            )
        if git(source, "status", "--porcelain", "--untracked-files=no"):
            problems.append("source tree has uncommitted changes")
    for problem in problems:
        print(problem)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> int:
    """Vendor the oracles or check them against the manifest.

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="checkout of pkpdutils")
    parser.add_argument("--check", action="store_true", help="re-hash, copy nothing")
    parser.add_argument(
        "--accept-changes",
        action="store_true",
        help="allow the set of vendored files to differ from the existing manifest",
    )
    args = parser.parse_args()
    if args.check:
        return check(args.source)
    if args.source is None:
        parser.error("--source is required to vendor")
    vendor(args.source, args.accept_changes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
