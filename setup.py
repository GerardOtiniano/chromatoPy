# setup.py
import sys
from setuptools import setup

# Prevent modulegraph from hitting the default recursion limit
sys.setrecursionlimit(10_000)

APP = ["src/chromatopy/__main__.py"]
OPTIONS = {
    # Don’t try to auto-scan every built-in or test dir
    "argv_emulation": True,
    "packages": ["chromatopy"],

    # Force-include the backends / C-ext libraries you know you need
    "includes": [
        "toga_cocoa",       # Toga’s macOS backend
        "toga.fonts",       # Toga font support
        "pybaselines",      # your baseline-fitter
        "numpy",            # and friends
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib.backends.backend_agg",
    ],

    # Exclude any modules you never actually import at runtime.
    "excludes": [
        "tkinter",          # you’re not using Tkinter here
        "unittest",         # test libraries
        "distutils",        # not needed at runtime
        "pip",              # ditto
        "setuptools",
        "email",            # stdlib email parser
        "xml",              # if you don’t parse XML
    ],
}

setup(
    app=APP,
    name="ChromatoPy",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)