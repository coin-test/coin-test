from nox_poetry import session

python_versions = ["3.10"]

@session(python=python_versions)
def tests(session):
    args = session.posargs or []
    args += ["--cov", "-m", "not e2e"]
    session.install("coverage[toml]", "pytest", "pytest-cov", ".")
    session.run("pytest", *args)
