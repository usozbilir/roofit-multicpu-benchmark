#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environment checker for the RooFit multi-CPU benchmark.

Checks:
  - Python modules: ROOT, pandas, matplotlib, psutil
  - External tool: lstopo (from hwloc)

If everything is available, exits with code 0.
If something is missing, prints detailed instructions and exits with code 1.
"""

import importlib
import shutil
import sys
import os
import platform

REQUIRED_MODULES = ["ROOT", "pandas", "matplotlib", "psutil"]
REQUIRED_TOOL = "lstopo"  # treated as mandatory


def print_header():
    print("=== RooFit multi-CPU benchmark environment check ===")
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.system()} {platform.release()}")
    print(f"Machine        : {platform.machine()}")
    print("")


def detect_env_type():
    """
    Very simple detection of environment type, only for tailoring messages.
    """
    in_conda = ("CONDA_DEFAULT_ENV" in os.environ) or ("CONDA_PREFIX" in os.environ)
    in_mamba = "MAMBA_DEFAULT_ENV" in os.environ
    if in_mamba:
        return "mamba"
    if in_conda:
        return "conda"
    return "unknown"


def check_python_modules():
    missing = []
    for mod in REQUIRED_MODULES:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


def check_lstopo():
    """
    Return True if lstopo is found in PATH, else False.
    """
    return shutil.which(REQUIRED_TOOL) is not None


def print_module_help(module_name, env_type):
    """
    Print installation hints + links for a specific Python module.
    """
    # Common links
    links = {
        "ROOT": "https://root.cern/install/",
        "pandas": "https://pandas.pydata.org/docs/getting_started/install.html",
        "matplotlib": "https://matplotlib.org/stable/install/index.html",
        "psutil": "https://psutil.readthedocs.io/",
    }

    print(f"- {module_name} is missing.")

    if module_name == "ROOT":
        print("  ROOT (PyROOT) is required for the benchmark.")
        print("  Please install ROOT and PyROOT using one of the official methods:")
        print("    - Official ROOT site: " + links["ROOT"])
        print("    - Conda (conda-forge):")
        print("        conda install -c conda-forge root")
        print("  Make sure that `python -c \"import ROOT\"` works in the same environment.")
    else:
        # pandas / matplotlib / psutil
        if env_type in ("conda", "mamba"):
            print("  You seem to be inside a conda/mamba environment.")
            print("  Example installation with conda-forge:")
            print(f"    conda install -c conda-forge {module_name.lower()}")
        else:
            print("  Example installation with pip:")
            print(f"    python -m pip install {module_name.lower()}")
        print(f"  Official docs: {links[module_name]}")

    print("")


def print_lstopo_help(os_name):
    """
    Print installation hints + links for lstopo (hwloc).
    lstopo is treated as mandatory for this benchmark.
    """
    print(f"- External tool '{REQUIRED_TOOL}' is missing (required).")
    print("  lstopo is part of the 'hwloc' package. Installation depends on your OS.\n")

    if os_name == "Linux":
        print("  On many Linux distributions, you can install it via your package manager, e.g.:")
        print("    - Debian/Ubuntu:  sudo apt-get install hwloc")
        print("    - Fedora/RHEL/CentOS:  sudo dnf install hwloc")
        print("    - openSUSE:  sudo zypper install hwloc")
    elif os_name == "Darwin":
        print("  On macOS, if you use Homebrew:")
        print("    brew install hwloc")
        print("  Make sure that 'lstopo' is in your PATH afterwards.")
    elif os_name == "Windows":
        print("  On Windows, lstopo is not installed by default.")
        print("  You can obtain hwloc (which includes lstopo) from the official project:")
        print("    https://www.open-mpi.org/projects/hwloc/")
        print("  After installation, ensure that 'lstopo.exe' is in your PATH.")
    else:
        print("  Please install 'hwloc' (which provides lstopo) for your platform.")
        print("  Official project page: https://www.open-mpi.org/projects/hwloc/")

    print("")


def main():
    print_header()

    os_name = platform.system()  # "Linux", "Darwin", "Windows", ...
    env_type = detect_env_type()

    # 1) Python modules
    missing_modules = check_python_modules()

    # 2) lstopo
    has_lstopo = check_lstopo()

    if not missing_modules and has_lstopo:
        print("All required components are available:")
        print("  - Python modules: ROOT, pandas, matplotlib, psutil")
        print(f"  - External tool: {REQUIRED_TOOL}")
        print("")
        print("You can now safely run the RooFit multi-CPU benchmark script.")
        sys.exit(0)

    # If we are here, something is missing
    print("Some required components are missing.\n")

    if missing_modules:
        print("Missing Python modules:")
        for mod in missing_modules:
            print_module_help(mod, env_type)

    if not has_lstopo:
        print("Missing external tools:")
        print_lstopo_help(os_name)

    print("Please install the missing components above and then re-run this check.")
    print("Once this script reports success, you can run the benchmark script.")
    sys.exit(1)


if __name__ == "__main__":
    main()
