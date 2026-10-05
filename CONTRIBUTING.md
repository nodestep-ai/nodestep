# Contributing

## Setup

nodestep uses [uv](https://docs.astral.sh/uv/). Install every extra and the dev tools once:

```bash
uv sync --locked --all-extras
```

After that, use `uv run --no-sync ...` so uv does not re-sync the environment. A plain `uv run` syncs without the extras and removes them.

## Checks

Run these before opening a pull request. CI runs the same commands.

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync ty check .
uv run --no-sync pytest -q
```

`src/nodestep_sandbox/static/` holds copies of `nodestep.css`, `nodestep-theme.js` and `nodestep-data.js` from [nodestep-stylesheet](https://github.com/nodestep-ai/nodestep-stylesheet). Change them there and copy them over. With nodestep-stylesheet checked out next to nodestep, `tests/design` checks that the copies match.

To build the documentation locally:

```bash
uv sync --locked --all-extras --group docs
uv run --no-sync mkdocs serve
```

`ty check .` skips `overrides/` and `tests/docs/`, which need the `docs` group. Check them with `uv run --no-sync ty check overrides tests/docs`. The docs theme script has a Node test: `node --test tests/docs/theme.test.ts`, with Node.js 22.18 or newer.

With nodeartifact checked out next to nodestep, the site includes its `docs/` folder as a section; without it, that section is left out. Its pages are listed by hand in the `nav` of `mkdocs.yml`, so a page added, renamed or removed in its `docs/` fails the strict build until the `nav` matches. Its doc changes go live on the next nodestep push to `main`, or when the Docs workflow is run by hand.

## Changes

- Every behavior change starts with a failing test.
- Public functions and classes get numpy-style docstrings.
- Every user-visible change adds a line under `## [Unreleased]` at the top of `CHANGELOG.md` (add the heading if it is not there), in the matching `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed` or `Security` section.

## Releases

1. Rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and add its link at the bottom of `CHANGELOG.md`.
2. Set `X.Y.Z` as `version` in `pyproject.toml` and in the alpha notes of `README.md` and `docs/index.md`, and update the version in `tests/test_packaging.py`, `tests/docs/test_readme.py` and `tests/docs/test_pages.py`.
3. Merge into `main`. When CI passes there, the release workflow checks that the changelog version matches `pyproject.toml`, builds the package, and creates the tag `vX.Y.Z` and a GitHub release with that changelog section as its notes and a link to the changelog. A `main` commit whose newest version is already released makes no release.
