"""Plugin docstring."""

__version__ = "0.1"
__author__ = "Psi4 Developer"

# Load Python modules
# Load C++ plugin
import os

from .pymodule import *

plugdir = os.path.split(os.path.abspath(__file__))[0]
# sofile = plugdir + '/' + os.path.split(plugdir)[1] + '.so'
# psi4.plugin_load(sofile)
