#!/usr/bin/env python

from io import open
from setuptools import setup, find_packages
import os

"""
:authors: Dmitrii Beregovoi
:license: BSD 3-Clause License, see LICENSE file
:copyright: (c) 2025 Dmitrii Beregovoi
"""

here = os.path.abspath(os.path.dirname(__file__))

def get_version():
    version_file = os.path.join(here, 'pltstat', '__init__.py')
    with open(version_file) as f:
        for line in f:
            if line.startswith('__version__'):
                return line.split('=')[-1].strip().strip('"')
    raise RuntimeError("Unable to find version string.")


with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def load_requirements():
    with open(os.path.join(here, "requirements.txt")) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# long_description = """Python library designed to facilitate
# the visualization of statistical data analysis. This library
# includes a variety of tools and methods to streamline data exploration,
# statistical computation, and graphical representation."""

version = get_version()


setup(
    name="pltstat",
    version=version,

    author="Dmitrii Beregovoi",
    author_email='dimaforth@gmail.com',

    description="A Python Library for Statistical Data Visualization",
    long_description=long_description,
    long_description_content_type="text/markdown",

    url="https://github.com/trojanskehesten/pltstat",
    download_url=f"https://github.com/trojanskehesten/pltstat/archive/refs/tags/v{version}.zip",

    license="BSD 3-Clause License, see LICENSE file",

    packages=["pltstat"],
    # packages=find_packages(),
    install_requires=load_requirements(),
    python_requires="~=3.12",
    include_package_data=True,  # Reading Manifest.in

    keywords = "matplotlib, statistics, visualization, dataanalysis",

    classifiers=[
        "Development Status :: 4 - Beta",  # "5 - Production/Stable"
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ]
)

# python setup.py sdist
# twine upload dist/*
