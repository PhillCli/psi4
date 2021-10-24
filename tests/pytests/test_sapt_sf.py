import pytest
from .utils import *

import psi4


def test_sapt_sf_rohf_cph():
    """test cp-rohf solver for SAPT-SF"""

    psi4.geometry("""
    0 2
    Li 0.000 0.000 -2.000
    --
    0 2
    Na 0.000 0.000 +2.000

    no_reorient
    no_com
    units angstrom
    symmetry c1
    """)

    psi4.set_options({
        'scf_type': 'df',
        'reference': 'rohf',
        'guess': 'sad',
        'basis': 'aug-cc-pvdz',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10,
        'maxiter': 100,
        'DO_ONLY_CPHF': True,
    })

    energy = psi4.energy('SF-SAPT')