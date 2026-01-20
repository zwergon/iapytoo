# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from pathlib import Path
import sys
import os
project = 'iapytoo'
copyright = '2026, Lecomte Jean-François'
author = 'Lecomte Jean-François'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    'sphinx_rtd_theme',
    'myst_parser',
    "sphinx.ext.autodoc"
]


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = "_static/images/logo_transparent.png"
html_favicon = "_static/images/icon.jpg"


napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Désactiver si tu n'utilises pas le format NumPy

# Pour que Sphinx trouve iapytoo
sys.path.insert(0, Path(__file__).parents[2].resolve().as_posix())
