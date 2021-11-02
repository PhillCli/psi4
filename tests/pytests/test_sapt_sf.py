import pytest
from .utils import compare_values

import psi4

# NOTE: results verified with finite-field custom code
li_na_simple = """
0 2
Li 0.000 0.000 -1.000
--
0 2
Na 0.000 0.000 +1.000

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

he_n_zuchowski2003 = """
0 1
He 0.000 0.000 0.000
--
0 4
N  0.000 0.000 7.000

units bohr
symmetry c1
no_reorient
no_com
"""

# note CN - Ne, ref: −16.41 cm^-1
# -6.040717e-06 ~= -13.26 cm^-1
reference_data = [(li_na_simple, -0.2807338409357377), (cn_ne_zuchowski2008, -6.0407170195471045e-05),
                  (he_n_zuchowski2003, 0.0)]


@pytest.mark.parametrize("geometry,reference", reference_data)
def test_sapt_sf_rohf_cphf(geometry, reference):
    """test cp-rohf solver for SAPT-SF"""

    psi4.geometry(geometry)

    psi4.set_options({
        'scf_type': 'pk',
        'reference': 'rohf',
        'guess': 'sad',
        'basis': 'aug-cc-pvdz',
        'e_convergence': 1e-10,
        'd_convergence': 1e-10,
        'maxiter': 200,
        'SF_SAPT_DO_ONLY_CPHF': True,
    })

    psi4.energy('SF-SAPT')
    assert compare_values(psi4.core.variable('SAPT IND20,R ENERGY'), reference, 6, "SAPT(ROHF) IND20,resp energy")
