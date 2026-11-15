"""Vendor the conformance oracles of `pkpdutils` into `testdata/oracles/`.

Copies the reference files the Python tests pin their numbers to, keeping their
relative paths, and writes `MANIFEST.json` with the sha256 of every file and the
commit of the `pkpdutils` repository they were copied from. It refuses a source
tree with uncommitted changes, since the recorded commit would not describe the
copied content.

With `--check` nothing is copied. The vendored files are re-hashed against the
manifest and, when `--source` is given, the source files against it as well and
the commit of the source against the recorded one. It exits non-zero on any
difference and names the files. Standard library only.
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
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(source: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def source_commit(source: Path) -> str:
    """Return the commit of a clean source tree, refuse a dirty one."""
    dirty = git(source, "status", "--porcelain")
    if dirty:
        sys.exit(
            f"refusing: {source} has uncommitted changes, the recorded commit "
            f"would not describe the copied files:\n{dirty}"
        )
    return git(source, "rev-parse", "HEAD")


def relative_paths(source: Path) -> list[str]:
    paths = list(FILES)
    for directory in DIRECTORIES:
        paths += [
            p.relative_to(source).as_posix()
            for p in (source / directory).rglob("*")
            if p.is_file()
        ]
    missing = [p for p in paths if not (source / p).is_file()]
    if missing:
        sys.exit(f"missing in source: {', '.join(missing)}")
    return sorted(paths)


def vendor(source: Path) -> None:
    commit = source_commit(source)
    paths = relative_paths(source)
    if ORACLE_DIR.exists():
        shutil.rmtree(ORACLE_DIR)
    files = {}
    for rel in paths:
        target = ORACLE_DIR / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, target)
        files[rel] = sha256(target)
    manifest = {"source_commit": commit, "files": files}
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"vendored {len(files)} files from pkpdutils {commit}")


def check(source: Path | None) -> int:
    manifest = json.loads(MANIFEST_PATH.read_text())
    recorded = manifest["files"]
    problems = []
    for rel, digest in recorded.items():
        path = ORACLE_DIR / rel
        if not path.is_file():
            problems.append(f"vendored file missing: {rel}")
        elif sha256(path) != digest:
            problems.append(f"vendored file differs from manifest: {rel}")
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
        if git(source, "status", "--porcelain"):
            problems.append("source tree has uncommitted changes")
    for problem in problems:
        print(problem)
    print("ok" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, help="checkout of pkpdutils")
    parser.add_argument("--check", action="store_true", help="re-hash, copy nothing")
    args = parser.parse_args()
    if args.check:
        return check(args.source)
    if args.source is None:
        parser.error("--source is required to vendor")
    vendor(args.source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
