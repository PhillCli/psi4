import psi4

import numpy as np

# from psi4.driver.p4util.solvers import davidson_solver
# from psi4.driver.procrouting.response.scf_products import SCFProducts

mol = psi4.geometry("""
0 3
N1 0.000    0.000     1.81585492519
H1 0.000    0.000     0.779654925192
--
0 1
Gh(N2) 0.000    0.000    -1.67671491821
Gh(H2) 0.000    0.000    -2.71291491821
units angstrom
no_reorient
symmetry c1
""")

psi4.set_options({"SAVE_JK": True})
psi4.set_options({"e_convergence": 1.0e-8, "d_convergence": 1.0e-8})
psi4.set_options({"reference": "rohf"})
psi4.set_options({"basis": "aug-cc-pvdz"})
e, wfn = psi4.energy("HF", mol=mol, return_wfn=True)
mol = psi4.geometry("""
0 1
Gh(N1) 0.000    0.000     1.81585492519
Gh(H1) 0.000    0.000     0.779654925192
--
0 3
N2     0.000    0.000    -1.67671491821
H2     0.000    0.000    -2.71291491821
units angstrom
no_reorient
symmetry c1
""")
e, wfn_B = psi4.energy("HF", mol=mol, return_wfn=True)

wfn.semicanonicalize()

nmo = wfn.nmopi().sum()
ndocc = wfn.doccpi().sum()
nsocc = wfn.soccpi().sum()
nvir = nmo - ndocc - nsocc
nrot = ndocc * nsocc + nsocc * nvir + ndocc * nvir

print(f"MO  : {nmo}")
print(f"DOCC: {ndocc}")
print(f"SOCC: {nsocc}")
print(f"NVIR: {nvir}")
print(f"NROT: {nrot}")

mints = psi4.core.MintsHelper(wfn_B.basisset())
V_B = mints.ao_potential()

Cdocc_npy = np.array(wfn.Cb_subset("AO", "OCC"))[:, :]
Csocc_npy = np.array(wfn.Ca_subset("AO", "OCC"))[:, ndocc:]
Cvirt_npy = np.array(wfn.Ca_subset("AO", "VIR"))[:, :]
Ctest_npy = np.array(wfn.Cb_subset("AO", "VIR"))
omega_mo_npy = np.zeros((ndocc + nsocc, nsocc + nvir))
V_B_npy = np.array(V_B)

# omega = V + 2J[D_docc] + J[D_socc]
jk = wfn.jk()
jk.set_do_K(False)
jk.print_header()
jk.C_clear()

nmo_B = wfn.nmopi().sum()
ndocc_B = wfn.doccpi().sum()
nsocc_B = wfn.soccpi().sum()
nvi_B = nmo - ndocc - nsocc
Cdocc_B_npy = np.array(wfn_B.Cb_subset("AO", "OCC"))[:, :]
Csocc_B_npy = np.array(wfn_B.Ca_subset("AO", "OCC"))[:, ndocc_B:]
# J[D_docc]
jk.C_left_add(psi4.core.Matrix.from_array(Cdocc_B_npy))
jk.C_right_add(psi4.core.Matrix.from_array(Cdocc_B_npy))
# J[D_socc]
jk.C_left_add(psi4.core.Matrix.from_array(Csocc_B_npy))
jk.C_right_add(psi4.core.Matrix.from_array(Csocc_B_npy))
# compute
jk.compute()
J_Ddocc = jk.J()[0].clone()
# _ = jk.K()[0].clone()
J_Dsocc = jk.J()[1].clone()
# omega = V + 2J[D_docc] + J[D_socc]
omega_ao_npy = V_B_npy.copy()
omega_ao_npy += 2 * np.array(J_Ddocc)
omega_ao_npy += np.array(J_Dsocc)
omega_ao = psi4.core.Matrix.from_array(omega_ao_npy, "Cdocc")

Cdocc = psi4.core.Matrix.from_array(Cdocc_npy, "Cdocc")
Csocc = psi4.core.Matrix.from_array(Csocc_npy, "Csocc")
Cvirt = psi4.core.Matrix.from_array(Cvirt_npy, "Cvirt")

# ROHF::Hx expected strucutre
# docc x socc | docc x virt
# socc x socc | socc x virt
# NOTE: socc x socc is always zero

omega_mo_docc_socc = psi4.core.triplet(Cdocc, omega_ao, Csocc, True, False, False)
omega_mo_docc_virt = psi4.core.triplet(Cdocc, omega_ao, Cvirt, True, False, False)
omega_mo_socc_virt = psi4.core.triplet(Csocc, omega_ao, Cvirt, True, False, False)
print(f"omega_mo_1.shape: {np.array(omega_mo_docc_socc).shape}")
print(f"omega_mo_2.shape: {np.array(omega_mo_docc_virt).shape}")
print(f"omega_mo_3.shape: {np.array(omega_mo_socc_virt).shape}")
print(f"omega_mo.shape: {omega_mo_npy.shape}")
# set the MO blocks
omega_mo_npy[:ndocc, :nsocc] = np.array(omega_mo_docc_socc)
omega_mo_npy[:ndocc, nsocc:] = np.array(omega_mo_docc_virt)
omega_mo_npy[ndocc:ndocc + nsocc, nsocc:] = np.array(omega_mo_socc_virt)
omega_mo = psi4.core.Matrix.from_array(omega_mo_npy, "Omega_B_MO")
# omega_mo = psi4.core.Matrix("", ndocc + nsocc, nsocc + nvir)
# omega_mo = psi4.core.triplet(Csocc, V, Ctest, True, False, False)

print(f"Cdocc.shape: {np.array(Cdocc).shape}")
print(f"Csocc.shape: {np.array(Csocc).shape}")
print(f"Cvirt.shape: {np.array(Cvirt).shape}")
print(f"V.nirrep(): {V_B.nirrep()}")
print(f"omega_mo.shape: {np.array(omega_mo).shape}")

print(wfn)
ret = wfn.cphf_solve(x_vec=[omega_mo], conv_tol=1e-9, max_iter=200, print_lvl=2)
t_ind_A = np.array(ret[0])
t_ia = t_ind_A[:ndocc, :nsocc]
t_ir = t_ind_A[:ndocc, nsocc:]
t_ar = t_ind_A[ndocc:ndocc + nsocc, nsocc:]
print(t_ia.shape)
print(t_ir.shape)
print(t_ar.shape)
print(np.einsum("ab,ab", t_ind_A, omega_mo))
# print(t_ind_A)
