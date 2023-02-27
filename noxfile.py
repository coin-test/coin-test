"""Nox sessions."""


import nox  # pyright: ignore
from nox_poetry import Session, session  # pyright: ignore


nox.options.sessions = "lint", "pyright", "typeguard", "tests"

package = "coin_test"
python_versions = ["3.10"]
locations = "src", "tests", "noxfile.py"


@session(python=python_versions)
def black(session: Session) -> None:
    """Run black code formatter."""
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@session(python=python_versions)
def lint(session: Session) -> None:
    """Lint using flake8."""
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-black",
        "flake8-bugbear",
        "flake8-docstrings",
        "flake8-import-order",
        "darglint",
    )
    session.run("flake8", *args)


@session(python=python_versions)
def pyright(session: Session) -> None:
    """Type checking using pyright."""
    args = session.posargs or locations
    args = list(args) + [
        "--project",
        ".nox_pyright_config.json",
        "--venv-path",
        ".nox/pyright-3-10",
    ]
    session.install("pyright", "pytest", "pytest-mock")
    session.run_always("poetry", "install", external=True)
    session.run("pyright", *args)


@session(python=python_versions)
def typeguard(session: Session) -> None:
    """Runtime type checking using Typeguard."""
    args = session.posargs or ["-m", "not (e2e or exclude_typeguard)"]
    session.install("pytest", "pytest-mock", "typeguard", ".")
    session.run("pytest", f"--typeguard-packages={package}", *args)


@session(python=python_versions)
def tests(session: Session) -> None:
    """Run the test suite."""
    args = session.posargs or ["--cov", "-m", "not e2e"]
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-mock", ".")
    session.run("pytest", *args)


@session(python=python_versions)
def coverage(session: Session) -> None:
    """Generate the coverage data."""
    session.install("coverage[toml]", "codecov")
    session.run("coverage", "xml", "--fail-under=0")


@session(python=python_versions)
def docs(session: Session) -> None:
    """Generate the docs."""
    session.install(
        "sphinx_autodoc_typehints",
        "sphinx-rtd-theme",
        "pandas",
        "croniter",
        "numpy",
        "sphinx",
    )
    session.run("sphinx-build", "docs", "_build", "-W", "--keep-going")
