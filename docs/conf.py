"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
# autodoc_mock_imports = ["numpy", "pandas", "croniter"]

project = "coin-test"
copyright = "2023, Olin SCOPE"
author = "Olin SCOPE"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = []

language = "python"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_logo = "_static/coin-test-logo-sphinx.png"

html_theme = "sphinx_rtd_theme"
# html_static_path = ["_static"]  # add this once static pages are necessary

html_theme_options = {
    "logo_only": True,
    "prev_next_buttons_location": "bottom",
    "collapse_navigation": True,
}
