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

All dependencies are managed by poetry. To add a new dependency, run `poetry add <package>`.
To add a new dev dependency, run `poetry add --group dev <package>`.
Dev package will not ship with the published version of `coin-test`.

Also note that since poetry installs all packages in a virtual environment,
locally we must also run `coin-test` through the same environment. This means
using `poetry run <cmd>` for each command (ex: `poetry run pytest`), or
activating the virtual environment with `poetry shell`.

## Logging

Logs are generated via Python's
[`logging` package](https://docs.python.org/3/howto/logging.html).

### When to Log

The large majority of logs in this library are of two main types: `info` and `debug`.
These logs help inform the user of processes and guide developers debugging the library.

The `info` logs are the most important logs, as these are displayed by default for the user.
These logs should be written with the user in mind and should only be used when the user
would care about what the library is doing. More specifically, `info` logs should be used when
an important process of interest to the user has an update. For example, when backtesting
begins, analysis begins, or a report is generated, this should be logged with the `info` type.
No objects should be dumped in an `info` log unless absolutely necessary.

The `debug` logs assist in informing a developer of the status of the library at a finer
granularity than `info` logs. As such, this can include the start and end of processes only
a developer would know or care about, like when analysis metrics are calculated. They can
also be used to give more information about a process already documented with `info` logs.
This can include, for example, logging information about strategies, synthetic data, and
other parameters at the beginning of a backtest. The `debug` logs can further be used to flag
important parts of a larger process.

More information on how logs should be created can be found in the official
[logging documentation](https://docs.python.org/3/howto/logging.html#when-to-use-logging)

### Configuring Logs

Each file with logs should have the following lines towards the top of the file:

```python
import logging
logger = logging.getLogger(__name__)
```

This logging instance should be used for all logs, e.g. `logger.info()` or
`logger.debug()`. No direct print statements should be used in conjunction with these
logs. The only exception to this is status bars, like tqdm. To incorporate these status
bars in harmony with logging, use the `tqdm.contrib.logging.logging_redirect_tqdm`
function as described in its
[documentation](https://tqdm.github.io/docs/contrib.logging/#tqdmcontriblogging).

## Pull Request Process

## Code Review Process

## Code of Conduct
