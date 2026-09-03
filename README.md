# aces-relaxations

[![Python 3.12+](https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyPI - Version](https://img.shields.io/pypi/v/aces-relaxations)](https://pypi.org/project/aces-relaxations/)
[![Tests](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/tests.yaml)
[![Code Quality](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml/badge.svg)](https://github.com/McDermottHealthAI/aces-relaxations/actions/workflows/code-quality-main.yaml)
[![Contributors](https://img.shields.io/github/contributors/McDermottHealthAI/aces-relaxations.svg)](https://github.com/McDermottHealthAI/aces-relaxations/graphs/contributors)
[![Pull Requests](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/McDermottHealthAI/aces-relaxations/pulls)
[![License](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](LICENSE)

Relaxations and structural properties for ACES task configurations, to map future-only task definitions onto reduced query languages.

## Quick start

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. To get set up:

```bash
git clone https://github.com/McDermottHealthAI/aces-relaxations.git
cd aces-relaxations
uv sync
uv run pre-commit install
```

Run the tests:

```bash
uv run pytest -v
```

See [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the full development guide (build system, testing
conventions, code style, PR workflow).

> [!WARNING]
> There is no folder in this repository for `data`. Datasets — public or private — must be stored
> outside the repository to avoid leaking sensitive data, bloating the repo, or over-fitting the
> project to a single data resource. The same applies to API keys, tokens, and credentials. Note
> that anything committed to `git` history can be recovered through the published repository even
> if removed from the current tree or pushed to a non-main branch; if something sensitive is
> accidentally committed, you need to
> [purge it from history](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository).

## Repository management

This repository lives within the
['McDermottHealthAI'](https://github.com/McDermottHealthAI) GitHub organization, which sets default
issue labels that propagate to new repos. All changes to `main` go through pull requests; see the
"Pull Request Workflow" section of [`CONTRIBUTORS.md`](CONTRIBUTORS.md) for the expected flow.
Versioning follows semantic versioning, managed through `git` tags (e.g., `git tag 0.0.1`) —
`setuptools-scm` reads the tag and stamps the package version automatically.
