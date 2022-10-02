import nox  # pyright: ignore
from nox_poetry import Session, session  # pyright: ignore


nox.options.sessions = "lint", "pyright", "typeguard", "tests"

package = "coin_test"
python_versions = ["3.10"]
locations = "src", "tests", "noxfile.py"


@session(python=python_versions)
def black(session: Session) -> None:
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@session(python=python_versions)
def lint(session: Session) -> None:
    args = session.posargs or locations
    session.install(
        "flake8",
        "flake8-annotations",
        "flake8-black",
        "flake8-bugbear",
        "flake8-import-order",
    )
    session.run("flake8", *args)


@session(python=python_versions)
def pyright(session: Session) -> None:
    args = session.posargs or locations
    session.install("pyright")
    session.run("pyright", *args)


@session(python=python_versions)
def tests(session: Session) -> None:
    args = session.posargs or ["--cov", "-m", "not e2e"]
    session.install("coverage[toml]", "pytest", "pytest-cov", "pytest-mock", ".")
    session.run("pytest", *args)


@session(python=python_versions)
def typeguard(session: Session) -> None:
    args = session.posargs or ["-m", "not e2e"]
    session.install("pytest", "pytest-mock", "typeguard", ".")
    session.run("pytest", f"--typeguard-packages={package}", *args)
