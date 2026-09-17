# Release and publishing guide

## Identity

Distribution: `aimtv-plex`. Command/import package: `aimtv`.
The only version declaration is `src/aimtv/__init__.py::__version__`;
setuptools reads it using `[tool.setuptools.dynamic]`.
Use Python-compatible versions such as `0.5.0a1`, `0.5.0rc1`, and `0.5.0`.
Never reuse a published version for different code or files.

## One-time PyPI authorization

A GitHub release does not itself register or publish a PyPI project. The repository
workflow uses PyPI Trusted Publishing (OIDC), not a long-lived API token.
The PyPI account that should own the project must add a **pending publisher** at:

https://pypi.org/manage/account/publishing/

Set these exact fields:

| Field | Value |
| --- | --- |
| PyPI project name | `aimtv-plex` |
| GitHub owner | `smilidon` |
| Repository | `aimtv-plex` |
| Workflow filename | `release.yml` |
| Environment name | `pypi` |

Do not enter `.github/workflows/release.yml` in the workflow filename field.
A pending publisher does not reserve the project name. The first successful upload
creates the project under the authorizing PyPI account. If the name is unavailable,
choose another distribution name and update metadata/docs before publishing.
For an existing project owned by you, add this publisher in that project's Publishing
settings instead of creating a pending publisher.

The workflow declares the GitHub environment `pypi`. Configure its deployment branch
policy to permit `master`, and optionally require owner approval for publication.
Never paste passwords, API tokens or two-factor recovery codes into issues or chat.

Official setup documentation:
https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
https://docs.pypi.org/trusted-publishers/using-a-publisher/

## Publishing a new version

1. Update `__version__` and add `docs/releases/<version>.md` with installation,
   changes, validation scope and known limitations. Update versioned README examples.
2. Push or merge the changes to `master`. A change to the version file triggers
   `.github/workflows/release.yml`. Only this repository's `master` can publish.
3. The release workflow tests/builds on Python 3.10 and 3.12, checks distribution
   metadata and smoke-tests the wheel in a fresh environment without the source tree.
4. Only after both jobs succeed, the workflow creates `v<version>` at the exact
   tested commit, uploads the Python wheel, source distribution and SHA256SUMS to a
   draft GitHub release, then publishes it. Alpha/beta/RC versions are prereleases.
5. A separate, least-privilege `pypi` job downloads those exact published assets,
   verifies their checksums, and uploads the distributions using Trusted Publishing.
   GitHub release success and PyPI publication success are separate results.

The workflow never replaces files in a published GitHub release or moves a tag.
Retries may complete an unpublished draft at the same commit. PyPI retries use the
original GitHub release assets, not newly rebuilt files, and skip already uploaded
files. Re-running from a different commit with the same version is rejected.

## Retrying PyPI after authorization

Configure the pending/trusted publisher first. In GitHub Actions, open the original
**Release** run and choose **Re-run failed jobs**. This reuses the original release
version and commit. Do not bump the version merely to retry a missing authorization.
The workflow also supports **Run workflow** on `master` when it still points to the
release commit; enter the exact version and enable `publish_pypi`.

Confirm the PyPI project/version page and its file hashes match the GitHub release
before advertising a PyPI installation command. The release workflow's final step
checks those hashes against the public PyPI JSON API after upload.

## Local development checks

```bash
python -m pip install -e ".[plex,dev]"
python -m pyflakes src/aimtv tests
python -m pytest -q
python -m build
python -m twine check dist/*
```

FFmpeg and ffprobe are required for the audio integration tests. Neither automated
CI nor wheel metadata validation establishes live Plex/GPU compatibility. Perform a
real installation and short end-to-end render before promoting an alpha to stable.
