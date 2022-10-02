import nox
from nox_poetry import session


nox.options.sessions = "lint", "tests"

python_versions = ["3.10"]
locations = "src", "tests", "noxfile.py"


@session(python=python_versions)
def lint(session):
    args = session.posargs or locations
    session.install("flake8", "flake8-black", "flake8-bugbear", "flake8-import-order")
    session.run("flake8", *args)


@session(python=python_versions)
def black(session):
    args = session.posargs or locations
    session.install("black")
    session.run("black", *args)


@session(python=python_versions)
def tests(session):
    args = session.posargs or []
    args += ["--cov", "-m", "not e2e"]
    session.install("coverage[toml]", "pytest", "pytest-cov", ".")
    session.run("pytest", *args)
