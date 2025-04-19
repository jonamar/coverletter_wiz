#!/usr/bin/env python
"""Setup script for the coverletter_wiz package."""

import os
from setuptools import setup

# Allow setup.py to be run from any path
os.chdir(os.path.normpath(os.path.join(os.path.abspath(__file__), os.pardir)))

if __name__ == "__main__":
    setup()
