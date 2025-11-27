# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'footrl' python package (exported copy).

This file is part of the attached `footrl` export and configures the
packaging to install the `footrl` top-level package that lives in this
exported tree.
"""

import os
import toml

from setuptools import setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file (tolerant to parse errors during editable installs)
try:
    EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))
except Exception as e:
    # Fall back to minimal defaults and warn the user. This helps when running
    # `pip install -e .` in environments where the extension.toml may contain
    # keys that a strict TOML parser in the build environment rejects.
    print("[WARN] Failed to parse extension.toml (continuing with defaults):", e)
    EXTENSION_TOML_DATA = {
        "package": {
            "author": "",
            "maintainer": "",
            "repository": "",
            "version": "0.0.0",
            "description": "",
            "keywords": [],
        }
    }

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# Installation operation
setup(
    name="footrl",
    packages=["footrl"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)