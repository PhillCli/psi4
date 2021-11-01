import pytest
from .utils import *

import psi4

# NOTE: results verified with finite-field custom code
li_na_simple = """
0 2
Li 0.000 0.000 -2.000
--
0 2
Na 0.000 0.000 +2.000

no_reorient
no_com
units angstrom
symmetry c1
"""

# NOTE: source http://dx.doi.org/10.1063/1.2968556
# TODO: add NH system as well
cn_ne_zuchowski2008 = """
0 2
C  0.000000 0.000000 +0.000000000
N  0.000000 0.000000 +2.210000000
--
0 1
Ne 0.000000 0.000000 -6.410228688

units bohr
symmetry c1
no_reorient
no_com
"""

reference_data = [
    (

    )
]
def test_sapt_sf_rohf_cphf():
    """test cp-rohf solver for SAPT-SF"""

    psi4.geometry(li_na_simple)

    psi4.set_options({
        'scf_type': 'direct',
        'reference': 'rohf',
        'guess': 'sad',
        'basis': 'aug-cc-pvdz',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10,
        'maxiter': 200,
        'SF_SAPT_DO_ONLY_CPHF': True,
    })

    energy = psi4.energy('SF-SAPT')
