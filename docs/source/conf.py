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
import os
import sys

sys.path.append(os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "Cornac"
copyright = "2023, Preferred.AI"
author = "Preferred.AI"

# The short X.Y version
version = "2.3"
# The full version, including alpha/beta/rc tags
release = "2.3.4"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_design",
    "myst_parser",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_title = "Cornac"
html_logo = "_static/logo.png"

html_theme_options = {
    # "switcher": {
    #     "json_url": "https://cornac.readthedocs.io/latest/_static/switcher.json",
    # }
    "external_links": [
        {
            "url": "https://cornac.preferred.ai",
            "name": "Official Site",
        },
        {
            "url": "https://preferred.ai",
            "name": "Preferred.AI",
        },
    ],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/preferredAI/cornac",
            "icon": "fa-brands fa-github",
        },
    ],
    "announcement": "https://raw.githubusercontent.com/PreferredAI/cornac/master/docs/announcement.html",
    "logo": {
        "text": "Cornac",
    },
    "pygment_light_style": "default",
    "pygment_dark_style": "github-dark",
    "secondary_sidebar_items": {
        "**": ["page-toc", "sourcelink"],
        "index": [],
        "models/index": [],
    },
}

html_sidebars = {
    "models/index": [],
    "index": [],
}

# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}
