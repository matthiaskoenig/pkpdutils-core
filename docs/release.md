# Release

How a release of `pkpdutils-core` is made. A tag makes both halves of it: the distribution on PyPI and the release of this repository, whose body is the release notes of that version. Neither is made by hand.

## The release notes

`release-notes/x.y.z.md` is the body of the release of `vx.y.z`. The job `publish` of `.github/workflows/release.yml` reads it before it uploads anything, so a tag whose notes were forgotten fails without spending the version number: a tag can be deleted and pushed again, a filename of PyPI cannot be taken back.

Write what a reader of the release needs: what changed, what it is worth where there are numbers, and what a caller has to do differently. The releases up to 0.2.1 were written by hand and carry no file here.

## The steps

1. branch off `develop`: `git switch -c release/x.y.z develop`
2. write `release-notes/x.y.z.md`
3. set `version` of `[workspace.package]` in `Cargo.toml` and refresh `Cargo.lock` (`cargo metadata > /dev/null`)
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

The tag then builds the five wheels and the sdist, uploads them to PyPI with trusted publishing and PEP 740 attestations, and creates the release `pkpdutils-core x.y.z` from `release-notes/x.y.z.md`. The release is made only once the upload succeeded, so a release of this repository never announces a version the index does not carry. A run repeated on a tag whose release exists already says so and changes nothing.
