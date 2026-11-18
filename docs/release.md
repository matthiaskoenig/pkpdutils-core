# Release

How a release of `pkpdutils-core` is made. A tag makes both halves of it: the distribution on PyPI and the release of this repository, whose body is the release notes of that version. Neither is made by hand.

## The release notes

`release-notes/x.y.z.md` is the body of the release of `vx.y.z`. The job `publish` of `.github/workflows/release.yml` reads it before it uploads anything, so a tag whose notes were forgotten fails without spending the version number: a tag can be deleted and pushed again, a filename of PyPI cannot be taken back.

Write what a reader of the release needs: what changed, what it is worth where there are numbers, and what a caller has to do differently. The releases up to 0.2.1 were written by hand and carry no file here.

## The steps

1. branch off `develop`: `git switch -c release/x.y.z develop`
2. write `release-notes/x.y.z.md`
3. set `version` of `[workspace.package]` in `Cargo.toml`, set the same version in the `version` of every path dependency on a crate of this workspace (`pkpdutils-abi` and `pkpdutils-units` in `crates/core/Cargo.toml`, `pkpdutils-abi` and `pkpdutils-core` in `crates/python/Cargo.toml`), set the same version in `js/package.json`, and refresh `Cargo.lock` (`cargo metadata --format-version 1 > /dev/null`). Those pins are what a consumer of the published crates resolves against, so `cargo metadata` fails until they match the new version; `js/prepack.mjs` refuses to pack a package whose version is not the version of the crates, and `js/test/package.test.mjs` compares both against what the module reports.
4. make sure everything passes: see [Checks](../README.md#checks)
5. push the branch, open the pull request against `develop` and merge it once the checks are green
6. tag the merged commit on `develop` and push the tag:

    ```bash
    git switch develop
    git pull
    git tag vx.y.z
    git push origin vx.y.z
    ```

    Check the version before pushing: the tag publishes to PyPI, and a version of the index cannot be taken back.

The tag then builds the five wheels and the sdist, uploads them to PyPI with trusted publishing and PEP 740 attestations, publishes the npm package of `js/` with trusted publishing and a provenance attestation, and creates the release `pkpdutils-core x.y.z` from `release-notes/x.y.z.md`. The release is made only once both uploads succeeded, so a release of this repository never announces a version an index does not carry. A run repeated on a tag whose release exists already says so and changes nothing, and the npm job skips a version npm already has, so a re-run of a failed release is harmless.

## The npm package, and its first publish

`js/` is the JavaScript host of the WebAssembly build of this kernel and is published as [`pkpdutils-core`](https://www.npmjs.com/package/pkpdutils-core) on npm, from the `npm` job of `.github/workflows/release.yml`, with the same version as the crates and the wheels.

npm trusted publishing is configured **per package** and a package has to exist before it can be configured, so the first publish is one manual step of the maintainer, once. It happens **before the tag of the release which first carries the package**, from the release branch after its pull request is green: the `github_release` job needs the `npm` job, which would fail on a package that does not exist yet and take the GitHub release of that tag with it. The job then finds the version on npm and skips its publish with a notice, which is what makes the order work.

```bash
cd js
npm login
node prepack.mjs
npm publish --access public
```

A tag which was pushed before that publish is not lost: publish by hand as above and re-run the failed workflow run, after which the `npm` job skips and the release of the tag is created.

Then set the trusted publisher on npmjs.com (Settings of the package, Trusted publisher: repository `matthiaskoenig/pkpdutils-core`, workflow `release.yml`), after which every release goes through the job and no token is held anywhere. The job itself refuses to publish unless the tag names the version of `js/package.json`, and it skips the publish with a notice when npm already carries that version.
