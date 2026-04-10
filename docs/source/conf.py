# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HERON'
copyright = '2025, Hepeng Li, Zhenlin Wang'
author = 'Hepeng Li, Zhenlin Wang'
release = 'v0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
import os
import sys

# Add paths for autodoc to find modules
sys.path.insert(0, os.path.abspath('../..'))  # heron package
sys.path.insert(0, os.path.abspath('../../case_studies/power'))  # powergrid package


extensions = [
    'myst_parser',
    'sphinxcontrib.mermaid',
    'sphinx.ext.autodoc',
    'sphinx_autodoc_typehints',
    'sphinx_design',
    'sphinx_copybutton',
    'sphinx.ext.githubpages',  # Creates .nojekyll file for GitHub Pages
]

myst_enable_extensions = [
    "colon_fence",  # allows ::: blocks
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    "logo": {
        "text": "HERON",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],  # Show navbar items
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_align": "left",
    "primary_sidebar_end": [],
    "secondary_sidebar_items": ["page-toc"],
    # "show_toc_level": 2,
    "navigation_depth": 2,
    "collapse_navigation": True,  # False enables expand/collapse functionality
    "show_nav_level": 0,  # Show only top level initially, expand on click
    "header_links_before_dropdown": 10,  # Show all items, no dropdown
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/yourusername/powergrid",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
}

# Use custom global toctree sidebar
html_sidebars = {
    "**": ["sidebar-nav-bs", "sidebar-ethical-ads"]
}

html_static_path = ['_static']

# Custom CSS files
html_css_files = [
    'css/custom.css',
]
