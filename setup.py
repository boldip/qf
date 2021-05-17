from setuptools import setup, find_packages
from os import path
from io import open
import re

here = path.abspath(path.dirname(__file__))

def read(*parts):
    with open(path.join(here, *parts), "r", encoding = "utf-8") as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match: return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name = "qf",
    version = find_version("src", "qf", "__init__.py"),
    description = "A package related to building quasi-fibration symmetries.",
    long_description = read("README.md"),
    long_description_content_type = "text/markdown",
    url = "https://github.com/boldip/qf",
    author = "Paolo Boldi",
    author_email = "paolo.boldi@unimi.it",
    classifiers = [
        "Development Status :: 1 - Beta",
        "Framework :: IPython",
        "Framework :: Jupyter",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.7",
    ],
    keywords = "graph automorphism fibration symmetry",
    package_dir = {"": "src"},
    packages = find_packages("src", exclude = ["tests"]),
    py_modules = ["scripts"],
    python_requires=">=3.7",
    install_requires = ["edist", "ipywidgets", "jupyter", "graphviz", "matplotlib", "networkx", "numpy", "pydot", "scipy", "sklearn", "zss"],
    extras_require = {
        "dev": [],
        "test": ["coverage", "codecov"],
    },
    #project_urls = {
    #    "Source": "https://github.com/boldip/qf"
    #    "Documentation": "https://github.com/boldip/qf"
    #},
)
