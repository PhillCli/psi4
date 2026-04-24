import pytest

import numpy as np
import psi4

from utils import compare_arrays

pytestmark = [pytest.mark.psi, pytest.mark.api]

def test_overlap_obs():
    h2o = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 101.5
        symmetry c1
    """)

    psi4.set_options({'basis': 'aug-cc-pvdz'})

    conv = psi4.core.BasisSet.build(h2o,'BASIS', psi4.core.get_global_option('BASIS'))

    wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option('BASIS'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    case1 = mints.ao_overlap()
    case2 = mints.ao_overlap(wfn.basisset(), wfn.basisset())

    assert psi4.compare_matrices(case1, case2, 10, "OVERLAP_TEST")  # TEST

def test_overlap_aux():
    h2o = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 101.5
        symmetry c1
    """)

    psi4.set_options({'basis': 'aug-cc-pvdz',
                      'df_basis_mp2':'aug-cc-pvdz-ri'})

    conv = psi4.core.BasisSet.build(h2o,'BASIS', psi4.core.get_global_option('BASIS'))
    aux = psi4.core.BasisSet.build(h2o,'DF_BASIS_MP2',"", "RIFIT", psi4.core.get_global_option('DF_BASIS_MP2'))

    wfn = psi4.core.Wavefunction.build(h2o, psi4.core.get_global_option('BASIS'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    tr = mints.ao_overlap(aux, aux).trace()

    assert psi4.compare_values(118, tr, 12, 'Test that diagonal elements of AO Overlap are 1.0')  # TEST

def test_export_ao_elec_dip_deriv():
    h2o = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 101.5
        symmetry c1
    """)

    rhf_e, wfn = psi4.energy('SCF/cc-pVDZ', molecule=h2o, return_wfn=True)

    mints = psi4.core.MintsHelper(wfn.basisset())

    natoms = h2o.natom()
    cart = ['_X', '_Y', '_Z']

    D = wfn.Da()
    D.add(wfn.Db())
    D_np = np.asarray(D)

    deriv1_mat = {}
    deriv1_np = {}

    MU_Gradient = np.zeros((3 * natoms, 3))

    for atom in range(natoms):
        deriv1_mat["MU_" + str(atom)] = mints.ao_elec_dip_deriv1(atom)
        for mu_cart in range(3):
            for atom_cart in range(3):
                map_key = "MU" + cart[mu_cart] + "_" + str(atom) + cart[atom_cart]
                deriv1_np[map_key] = np.asarray(deriv1_mat["MU_" + str(atom)][3 * mu_cart + atom_cart])
                MU_Gradient[3 * atom + atom_cart, mu_cart] += np.einsum("uv,uv->", deriv1_np[map_key], D_np)

    PSI4_MU_Grad = mints.dipole_grad(D)
    G_python_MU_mat = psi4.core.Matrix.from_array(MU_Gradient)
    assert psi4.compare_matrices(PSI4_MU_Grad, G_python_MU_mat, 10, "DIPOLE_GRADIENT_TEST")  # TEST

def test_export_ao_overlap_half_deriv():
    h2o = psi4.geometry("""
        O
        H 1 1.0
        H 1 1.0 2 101.5
        symmetry c1
    """)

    rhf_e, wfn = psi4.energy('SCF/cc-PVDZ', molecule=h2o, return_wfn=True)
    C = wfn.Ca_subset("AO", "ALL")

    mints = psi4.core.MintsHelper(wfn.basisset())

    natoms = h2o.natom()
    cart = ['_X', '_Y', '_Z']

    deriv1_mat = {}
    deriv1_np = {}

    # Get total overlap derivative integrals along with both left and right half-derivative integrals
    for atom in range(natoms):
        deriv1_mat["S_LEFT_HALF_" + str(atom)] = mints.mo_overlap_half_deriv1("LEFT", atom, C, C)
        deriv1_mat["S_RIGHT_HALF_" + str(atom)] = mints.mo_overlap_half_deriv1("RIGHT", atom, C, C)
        deriv1_mat["S_" + str(atom)] = mints.mo_oei_deriv1("OVERLAP", atom, C, C)
        for atom_cart in range(3):
            map_key1 = "S_LEFT_HALF_" + str(atom) + cart[atom_cart]
            map_key2 = "S_RIGHT_HALF_" + str(atom) + cart[atom_cart]
            map_key3 = "S_" + str(atom) + cart[atom_cart]
            deriv1_np[map_key1] = np.asarray(deriv1_mat["S_LEFT_HALF_" + str(atom)][atom_cart])
            deriv1_np[map_key2] = np.asarray(deriv1_mat["S_RIGHT_HALF_" + str(atom)][atom_cart])
            deriv1_np[map_key3] = np.asarray(deriv1_mat["S_" + str(atom)][atom_cart])

            # Test (S_ii)^x = 2 * < i^x | i >
            assert compare_arrays(deriv1_np[map_key1].diagonal(), deriv1_np[map_key3].diagonal()/2)

            # Test (S_ii)^x = 2 * < i | i^x >
            assert compare_arrays(deriv1_np[map_key2].diagonal(), deriv1_np[map_key3].diagonal()/2)

            # Test (S_ij)^x = < i^x | j > + < j^x | i >
            assert compare_arrays(deriv1_np[map_key1] + deriv1_np[map_key1].transpose(), deriv1_np[map_key3])

            # Test (S_ij)^x = < i^x | j > + < i | j^x >
            assert compare_arrays(deriv1_np[map_key1] + deriv1_np[map_key2], deriv1_np[map_key3])

def test_ao_erf_eri_mixed_basis():
    he = psi4.geometry("""
        He 0 0 0
        symmetry c1
    """)

    psi4.set_options({'basis': '3-21G'})
    orb = psi4.core.BasisSet.build(he, 'BASIS', psi4.core.get_global_option('BASIS'))
    aux = psi4.core.BasisSet.build(he, 'DF_BASIS_MP2', '', 'RIFIT', psi4.core.get_global_option('BASIS'))
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(orb)

    # Theory-reduction: omega=0 -> erf(0)/r = 0, NOT 1/r.
    # The erf attenuation at omega=0 is 0, so LR ERIs should be zero.
    erf_0 = np.array(mints.ao_erf_eri(0.0, aux, zero, orb, orb))
    assert np.allclose(erf_0, 0.0, atol=1e-14)

    # Mixed-basis consistency: ao_erf_eri vs ao_eri_omega alias
    erf_02 = np.array(mints.ao_erf_eri(0.2, aux, zero, orb, orb))
    omega_02 = np.array(mints.ao_eri_omega(0.2, aux, zero, orb, orb))
    assert np.allclose(erf_02, omega_02, rtol=1e-14, atol=1e-14)

    # Shape check for squeezed 3-index object (naux, nbf, nbf)
    assert erf_02.shape == (aux.nbf(), 1, orb.nbf(), orb.nbf())

    # Physical attenuation: omega>0 should suppress long-range contributions
    # relative to standard Coulomb (which is much larger for diffuse functions).
    # Compare diagonal metric element: (P|0|P)^{LR} < (P|0|P)^{Coulomb}
    coulomb_metric = np.array(mints.ao_eri(aux, zero, aux, zero))
    lr_metric = np.array(mints.ao_erf_eri(0.2, aux, zero, aux, zero))
    diag_coulomb = np.diag(coulomb_metric[:, 0, :, 0].reshape(aux.nbf(), aux.nbf()))
    diag_lr = np.diag(lr_metric[:, 0, :, 0].reshape(aux.nbf(), aux.nbf()))
    assert np.all(diag_lr < diag_coulomb)

    # Mixed-basis vs default-basis consistency for same basis
    erf_default = np.array(mints.ao_erf_eri(0.2))
    erf_mixed = np.array(mints.ao_erf_eri(0.2, orb, orb, orb, orb))
    assert np.allclose(erf_default, erf_mixed, rtol=1e-10, atol=1e-10)
