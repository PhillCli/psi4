#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2024 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
"""Module to facilitate quantum chemical computations on chemical
databases. Contains Molecule class and physical constants from psi4 suite.

"""

__version__ = "0.4"
__author__ = "Lori A. Burns"

# Load Python modules
import sys

from . import align, cfour, gradparse, hessparse, jajo, orca, vib
from .basislist import *
from .constants import constants
from .dbproc import *
from .dbwrap import DB4, Database  # DatabaseWrapper, ReactionDatum, Reagent, Reaction
from .libmintsbasisset import BasisSet
from .libmintsmolecule import LibmintsMolecule
from .libmintspointgrp import PointGroup, SymmetryOperation
from .molecule import Molecule, compute_atom_map
from .options import *

# Load items that are useful to access from an input file
from .psiutil import *
from .qcformat import *
from .testing import *
from .util import *
from .vib import compare_vibinfos
