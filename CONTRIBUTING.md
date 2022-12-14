# Contributing

## Development Environment

### Setup

First, ensure you have the following dependencies:

- `Python 3.10` or greater
- `poetry`
- `nox`, `nox-poetry`
- `pre-commit`

Nox and pre-commit can be installed with:

```bash
pip install --user --upgrade nox nox-poetry pre-commit
```

Poetry can be installed by following the instructions [here](https://python-poetry.org/docs/).

Next, clone the repo, `cd` into the repo and run `poetry install` and `pre-commit install`. All other
required packages will be automatically installed in virtual environment `.venv`.

### Linting

This project uses `black` for formatting, `flake8` (with a variety of
extensions) for linting, and `pyright` for type checking. `flake8` and `pyright`
are both configured by config files in the project root - ensure your IDE is
using these configs.

Formatting is not automatically run, but can be executed with `nox -s black`.

Linting is ran as part of the test suite, on `git commit` (given pre-commit is
installed) and on push to remote.

### Tests

Tests are stored in the `tests` directory and use the `pytest` framework. Run
the test suite with `nox` (or `nox -r`, for slightly more speed). This will
execute linting, the test suite with coverage, and the test suite with runtime
type checking.

Tests are run on push to remote.

### Dependency Management

All dependencies are managed by poetry. To add a new dependency, run `poetry add <package>`. To add a new dev dependency, run `poetry add --group dev <package>`.
Dev package will not ship with the published version of `coin-test`.

Also note that since poetry installs all packages in a virtual environment,
locally we must also run `coin-test` through the same environment. This means
using `poetry run <cmd>` for each command (ex: `poetry run pytest`), or
activating the virtual environment with `poetry shell`.

## Pull Request Process

## Code Review Process

## Code of Conduct
