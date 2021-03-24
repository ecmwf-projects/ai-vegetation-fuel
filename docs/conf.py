# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "smos-fuel"
copyright = "2021, Wikilimo"
author = "Wikilimo"

# The full version, including alpha/beta/rc tags
release = "0.1"

# -- specify the master doc, otherwise the build at read the docs fails ------
master_doc = "index"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = True

# Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "logs",
    "config",
    "cache",
    "__init__",
]


# -- autoapi configuration ---------------------------------------------------
autoapi_dirs = ["../src"]
autoapi_file_patterns = ["*.py"]
autoapi_ignore = ["*logs*", "__pycache__", "__init__*"]
autoapi_member_order = "bysource"
autoapi_keep_files = False
autoapi_add_toctree_entry = False
autoapi_options = [
    "members",
    "inherited-members",
    "special-members",
    "show-inheritance",
    "special-members",
    "imported-members",
    "show-inheritance-diagram",
]

# -- Turn on sphinx.ext.autosummary ------------------------------------------
autosummary_generate = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
