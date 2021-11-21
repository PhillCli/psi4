#
# @BEGIN LICENSE
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2021 The Psi4 Developers.
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
import time
from pprint import pformat
from typing import Tuple

import numpy as np
from collections import OrderedDict

from psi4 import core
from psi4.driver.p4util.exceptions import *
from psi4.driver.p4util import solvers

from .sapt_util import print_sapt_var

__all__ = ["compute_first_order_sapt_sf", "compute_cphf_induction"]


def _sf_compute_JK(jk, Cleft, Cright, rotation=None):
    """
    A specialized JK computer class for terms that arrise from SF-SAPT.

    The density is computed as (Cl_mu,i rotation_ij Cr_nu,j) where the rotation
    is an arbitrary perturbation on the density.
    """

    # Handle both list and single value input
    return_single = False
    if not isinstance(Cleft, (list, tuple)):
        Cleft = [Cleft]
        return_single = True
    if not isinstance(Cright, (list, tuple)):
        Cright = [Cright]
        return_single = True
    if (not isinstance(rotation, (list, tuple))) and (rotation is not None):
        rotation = [rotation]
        return_single = True

    if len(Cleft) != len(Cright):
        raise ValidationError("Cleft list is not the same length as Cright list")

    jk.C_clear()

    zero_append = []
    num_compute = 0

    for num in range(len(Cleft)):
        Cl = Cleft[num]
        Cr = Cright[num]

        if (Cr.shape[1] == 0) or (Cl.shape[1] == 0):
            zero_append.append(num)
            continue

        if (rotation is not None) and (rotation[num] is not None):
            mol = Cl.shape[1]
            mor = Cr.shape[1]

            if (rotation[num].shape[0] != mol) or (rotation[num].shape[1] != mor):
                raise ValidationError("_sf_compute_JK: Tensor size does not match Cl (%d) /Cr (%d) : %s" %
                                      (mol, mor, str(rotation[num].shape)))

            # Figure out the small MO index to contract to
            if mol < mor:
                Cl = np.dot(Cl, rotation[num])
            else:
                Cr = np.dot(Cr, rotation[num].T)

        Cl = core.Matrix.from_array(Cl)
        Cr = core.Matrix.from_array(Cr)

        jk.C_left_add(Cl)
        jk.C_right_add(Cr)
        num_compute += 1

    jk.compute()

    J_list = []
    K_list = []
    for num in range(num_compute):
        J_list.append(np.array(jk.J()[num]))
        K_list.append(np.array(jk.K()[num]))

    jk.C_clear()

    nbf = J_list[0].shape[0]
    zero_mat = np.zeros((nbf, nbf))
    for num in zero_append:
        J_list.insert(num, zero_mat)
        K_list.insert(num, zero_mat)

    if return_single:
        return J_list[0], K_list[0]
    else:
        return J_list, K_list


def _chain_dot(*dot_list):
    """
    A simple chain dot function unpacked from *args.
    """
    result = dot_list[0]
    for x in range(len(dot_list) - 1):
        result = np.dot(result, dot_list[x + 1])
    return result


def compute_first_order_sapt_sf(dimer, jk, wfn_A, wfn_B, do_print=True):
    """
    Computes Elst and Spin-Flip SAPT0 for ROHF wavefunctions
    """

    if do_print:
        core.print_out("\n  ==> Preparing SF-SAPT Data Cache <== \n\n")
        jk.print_header()

    ### Build intermediates

    # Pull out Wavefunction A quantities
    ndocc_A = wfn_A.doccpi().sum()
    nsocc_A = wfn_A.soccpi().sum()

    Cocc_A = np.asarray(wfn_A.Ca_subset("AO", "OCC"))
    Ci = Cocc_A[:, :ndocc_A]
    Ca = Cocc_A[:, ndocc_A:]
    Pi = np.dot(Ci, Ci.T)
    Pa = np.dot(Ca, Ca.T)

    mints = core.MintsHelper(wfn_A.basisset())
    V_A = mints.ao_potential()

    # Pull out Wavefunction B quantities
    ndocc_B = wfn_B.doccpi().sum()
    nsocc_B = wfn_B.soccpi().sum()

    Cocc_B = np.asarray(wfn_B.Ca_subset("AO", "OCC"))
    Cj = Cocc_B[:, :ndocc_B]
    Cb = Cocc_B[:, ndocc_B:]
    Pj = np.dot(Cj, Cj.T)
    Pb = np.dot(Cb, Cb.T)

    mints = core.MintsHelper(wfn_B.basisset())
    V_B = mints.ao_potential()

    # Pull out generic quantities
    S = np.asarray(wfn_A.S())

    intermonomer_nuclear_repulsion = dimer.nuclear_repulsion_energy()
    intermonomer_nuclear_repulsion -= wfn_A.molecule().nuclear_repulsion_energy()
    intermonomer_nuclear_repulsion -= wfn_B.molecule().nuclear_repulsion_energy()

    num_el_A = (2 * ndocc_A + nsocc_A)
    num_el_B = (2 * ndocc_B + nsocc_B)

    ### Build JK Terms
    if do_print:
        core.print_out("\n  ==> Computing required JK matrices <== \n\n")

    # Writen so that we can reorganize order to save on DF-JK cost.
    pairs = [("ii", Ci, None, Ci), ("ij", Ci, _chain_dot(Ci.T, S, Cj), Cj), ("jj", Cj, None, Cj), ("aa", Ca, None, Ca),
             ("aj", Ca, _chain_dot(Ca.T, S, Cj), Cj), ("ib", Ci, _chain_dot(Ci.T, S, Cb), Cb), ("bb", Cb, None, Cb),
             ("ab", Ca, _chain_dot(Ca.T, S, Cb), Cb)]

    # Reorganize
    names = [x[0] for x in pairs]
    Cleft = [x[1] for x in pairs]
    rotations = [x[2] for x in pairs]
    Cright = [x[3] for x in pairs]

    tmp_J, tmp_K = _sf_compute_JK(jk, Cleft, Cright, rotations)

    J = {key: val for key, val in zip(names, tmp_J)}
    K = {key: val for key, val in zip(names, tmp_K)}

    ### Compute Terms
    if do_print:
        core.print_out("\n  ==> Computing Spin-Flip Exchange and Electrostatics <== \n\n")

    w_A = V_A + 2 * J["ii"] + J["aa"]
    w_B = V_B + 2 * J["jj"] + J["bb"]

    h_Aa = V_A + 2 * J["ii"] + J["aa"] - K["ii"] - K["aa"]
    h_Ab = V_A + 2 * J["ii"] + J["aa"] - K["ii"]

    h_Ba = V_B + 2 * J["jj"] + J["bb"] - K["jj"]
    h_Bb = V_B + 2 * J["jj"] + J["bb"] - K["jj"] - K["bb"]

    ### Build electrostatics

    # socc/socc term
    two_el_repulsion = np.vdot(Pa, J["bb"])
    attractive_a = np.vdot(V_A, Pb) * nsocc_A / num_el_A
    attractive_b = np.vdot(V_B, Pa) * nsocc_B / num_el_B
    nuclear_repulsion = intermonomer_nuclear_repulsion * nsocc_A * nsocc_B / (num_el_A * num_el_B)
    elst_abab = two_el_repulsion + attractive_a + attractive_b + nuclear_repulsion

    # docc/socc term
    two_el_repulsion = np.vdot(Pi, J["bb"])
    attractive_a = np.vdot(V_A, Pb) * ndocc_A / num_el_A
    attractive_b = np.vdot(V_B, Pi) * nsocc_B / num_el_B
    nuclear_repulsion = intermonomer_nuclear_repulsion * ndocc_A * nsocc_B / (num_el_A * num_el_B)
    elst_ibib = 2 * (two_el_repulsion + attractive_a + attractive_b + nuclear_repulsion)

    # socc/docc term
    two_el_repulsion = np.vdot(Pa, J["jj"])
    attractive_a = np.vdot(V_A, Pj) * nsocc_A / num_el_A
    attractive_b = np.vdot(V_B, Pa) * ndocc_B / num_el_B
    nuclear_repulsion = intermonomer_nuclear_repulsion * nsocc_A * ndocc_B / (num_el_A * num_el_B)
    elst_jaja = 2 * (two_el_repulsion + attractive_a + attractive_b + nuclear_repulsion)

    # docc/docc term
    two_el_repulsion = np.vdot(Pi, J["jj"])
    attractive_a = np.vdot(V_A, Pj) * ndocc_A / num_el_A
    attractive_b = np.vdot(V_B, Pi) * ndocc_B / num_el_B
    nuclear_repulsion = intermonomer_nuclear_repulsion * ndocc_A * ndocc_B / (num_el_A * num_el_B)
    elst_ijij = 4 * (two_el_repulsion + attractive_a + attractive_b + nuclear_repulsion)

    elst = elst_abab + elst_ibib + elst_jaja + elst_ijij
    # print(print_sapt_var("Elst,10", elst))

    ### Start diagonal exchange

    exch_diag = 0.0
    exch_diag -= np.vdot(Pj, 2 * K["ii"] + K["aa"])
    exch_diag -= np.vdot(Pb, K["ii"])
    exch_diag -= np.vdot(_chain_dot(Pi, S, Pj), (h_Aa + h_Ab + h_Ba + h_Bb))
    exch_diag -= np.vdot(_chain_dot(Pa, S, Pj), (h_Aa + h_Ba))
    exch_diag -= np.vdot(_chain_dot(Pi, S, Pb), (h_Ab + h_Bb))

    exch_diag += 2.0 * np.vdot(_chain_dot(Pj, S, Pi, S, Pb), w_A)
    exch_diag += 2.0 * np.vdot(_chain_dot(Pj, S, Pi, S, Pj), w_A)
    exch_diag += np.vdot(_chain_dot(Pb, S, Pi, S, Pb), w_A)
    exch_diag += np.vdot(_chain_dot(Pj, S, Pa, S, Pj), w_A)

    exch_diag += 2.0 * np.vdot(_chain_dot(Pi, S, Pj, S, Pi), w_B)
    exch_diag += 2.0 * np.vdot(_chain_dot(Pi, S, Pj, S, Pa), w_B)
    exch_diag += np.vdot(_chain_dot(Pi, S, Pb, S, Pi), w_B)
    exch_diag += np.vdot(_chain_dot(Pa, S, Pj, S, Pa), w_B)

    exch_diag -= 2.0 * np.vdot(_chain_dot(Pi, S, Pj), K["ij"])
    exch_diag -= 2.0 * np.vdot(_chain_dot(Pa, S, Pj), K["ij"])
    exch_diag -= 2.0 * np.vdot(_chain_dot(Pi, S, Pb), K["ij"])

    exch_diag -= np.vdot(_chain_dot(Pa, S, Pj), K["aj"])
    exch_diag -= np.vdot(_chain_dot(Pi, S, Pb), K["ib"])
    # print(print_sapt_var("Exch10,offdiagonal", exch_diag))

    ### Start off-diagonal exchange

    exch_offdiag = 0.0
    exch_offdiag -= np.vdot(Pb, K["aa"])
    exch_offdiag -= np.vdot(_chain_dot(Pa, S, Pb), (h_Aa + h_Bb))
    exch_offdiag += np.vdot(_chain_dot(Pa, S, Pj), K["bb"])
    exch_offdiag += np.vdot(_chain_dot(Pi, S, Pb), K["aa"])

    exch_offdiag += 2.0 * np.vdot(_chain_dot(Pj, S, Pa, S, Pb), w_A)
    exch_offdiag += np.vdot(_chain_dot(Pb, S, Pa, S, Pb), w_A)

    exch_offdiag += 2.0 * np.vdot(_chain_dot(Pi, S, Pb, S, Pa), w_B)
    exch_offdiag += np.vdot(_chain_dot(Pa, S, Pb, S, Pa), w_B)

    exch_offdiag -= 2.0 * np.vdot(_chain_dot(Pa, S, Pb), K["ij"])
    exch_offdiag -= 2.0 * np.vdot(_chain_dot(Pa, S, Pb), K["ib"])
    exch_offdiag -= 2.0 * np.vdot(_chain_dot(Pa, S, Pj), K["ab"])
    exch_offdiag -= 2.0 * np.vdot(_chain_dot(Pa, S, Pj), K["ib"])

    exch_offdiag -= np.vdot(_chain_dot(Pa, S, Pb), K["ab"])
    # print(print_sapt_var("Exch10,off-diagonal", exch_offdiag))
    # print(print_sapt_var("Exch10(S^2)", exch_offdiag + exch_diag))

    ret_values = OrderedDict({
        "Elst10": elst,
        "Exch10(S^2) [diagonal]": exch_diag,
        "Exch10(S^2) [off-diagonal]": exch_offdiag,
        "Exch10(S^2) [highspin]": exch_offdiag + exch_diag,
    })

    return ret_values


def compute_cphf_induction(cache, jk, maxiter: int = 100, conv: float = 1e-6) -> Tuple[OrderedDict, OrderedDict]:
    """
    Solve the CP-ROHF for SAPT induction amplitudes.
    """
    # out of omega_A_ao & omega_B_ao
    # rhs_A -> should be MO transformation of omega_B (electrostatic potential of B monomer felt by mon A)
    # rhs_B -> should be MO transformation of omega_A (electrostatic potential of A monomer felt by mon B)

    wfn_A = cache["wfn_A"]
    wfn_B = cache["wfn_B"]
    # Pull out Wavefunction A quantities
    ndocc_A = wfn_A.doccpi().sum()
    nsocc_A = wfn_A.soccpi().sum()

    # i,j - core    (docc)
    # a,b - active  (socc)
    # r,s - virtual (virt)

    # Pull out Wavefunction B quantities
    ndocc_B = wfn_B.doccpi().sum()
    nsocc_B = wfn_B.soccpi().sum()

    # grab alfa & beta orbitals
    C_alpha_A = wfn_A.Ca_subset("AO", "OCC")
    C_beta_A = wfn_A.Cb_subset("AO", "OCC")
    C_alpha_B = wfn_B.Ca_subset("AO", "OCC")
    C_beta_B = wfn_B.Cb_subset("AO", "OCC")

    # grab virtual orbitals
    C_alpha_vir_A = wfn_A.Ca_subset("AO", "VIR")
    C_beta_vir_A = wfn_A.Cb_subset("AO", "VIR")
    C_alpha_vir_B = wfn_B.Ca_subset("AO", "VIR")
    C_beta_vir_B = wfn_B.Cb_subset("AO", "VIR")

    nbf, nvirt_A = C_alpha_vir_A.np.shape
    nbf, nvirt_B = C_alpha_vir_B.np.shape

    print(f"{ndocc_A=}")
    print(f"{nsocc_A=}")
    print(f"{nvirt_A=}")

    print(f"{ndocc_B=}")
    print(f"{nsocc_B=}")
    print(f"{nvirt_B=}")

    print("")
    print(f"{wfn_A.Cb_subset('AO', 'ALL').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'ACTIVE').np.shape= }")
    print(f"{wfn_A.Cb_subset('AO', 'FROZEN').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'OCC').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'VIR').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'FROZEN_OCC').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'ACTIVE_OCC').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'ACTIVE_VIR').np.shape=}")
    print(f"{wfn_A.Cb_subset('AO', 'FROZEN_VIR').np.shape=}")
    print("")
    print(f"{C_alpha_A.np.shape=}")
    print(f"{C_alpha_vir_A.np.shape=}")
    print(f"{cache['omega_B_ao'].np.shape=}")
    print("")
    print(f"{C_beta_A.np.shape=}")
    print(f"{C_beta_vir_A.np.shape=}")
    print(f"{cache['omega_B_ao'].np.shape=}")

    # first transfrom into MO in spin-blocks
    rhs_A_alpha = core.triplet(C_alpha_A, cache["omega_B_ao"], C_alpha_vir_A, True, False, False)
    rhs_A_beta = core.triplet(C_beta_A, cache["omega_B_ao"], C_beta_vir_A, True, False, False)

    # then retrive spin_blocks
    # omega_alpha = |omega_ar|
    #               |--------|
    #               |omega_ir|
    # NOTE: (ar, ai) or (ai, ar) for beta blocks?
    # omega_beta  = |omega_ar|
    #               |--------|
    #               |omega_ai|

    # NOTE: sanity check, if we got the ordering within spin-blocks right
    omega_ar_1 = rhs_A_beta.np[:, :nvirt_A]
    omega_ar_2 = rhs_A_alpha.np[:ndocc_A, :nvirt_A]
    print(f"{np.allclose(omega_ar_1, omega_ar_2)=}")

    rhs_A = core.Matrix(nsocc_A + ndocc_A, nsocc_A + nvirt_A)
    # omega_ai
    print(f"{rhs_A.np[:ndocc_A, :nsocc_A].shape=}")
    print(f"{rhs_A_beta.np.shape=}")
    print(f"{rhs_A_beta.np[:, nvirt_A:].shape=}")
    rhs_A.np[:ndocc_A, :nsocc_A] = rhs_A_beta.np[:, nvirt_A:]
    # omega_ar
    rhs_A.np[:ndocc_A, nsocc_A:] = rhs_A_beta.np[:, :nvirt_A]
    # omega_ir
    rhs_A.np[ndocc_A:, nsocc_A:] = rhs_A_alpha.np[ndocc_A:, :]
    # omega_ii
    rhs_A.np[ndocc_A:, :nsocc_A] = np.zeros((nsocc_A, nsocc_A))

    # take care of rhs_B
    rhs_B_alpha = core.triplet(C_alpha_B, cache["omega_A_ao"], C_alpha_vir_B, True, False, False)
    rhs_B_beta = core.triplet(C_beta_B, cache["omega_A_ao"], C_beta_vir_B, True, False, False)
    # NOTE: sanity check, if we got the ordering within spin-blocks right
    omega_bs_1 = rhs_B_beta.np[:, :nvirt_B]
    omega_bs_2 = rhs_B_alpha.np[:ndocc_B, :nvirt_B]
    print(f"{np.allclose(omega_bs_1, omega_bs_2)=}")

    rhs_B = core.Matrix(nsocc_B + ndocc_B, nsocc_B + nvirt_B)
    # omega_bj
    print(f"{rhs_B.np[:ndocc_B, :nsocc_B].shape=}")
    print(f"{rhs_B_beta.np.shape=}")
    print(f"{rhs_B_beta.np[:, nvirt_A:].shape=}")
    rhs_B.np[:ndocc_B, :nsocc_B] = rhs_B_beta.np[:, nvirt_B:]
    # omega_bs
    rhs_B.np[:ndocc_B, nsocc_B:] = rhs_B_beta.np[:, :nvirt_B]
    # omega_js
    rhs_B.np[ndocc_B:, nsocc_B:] = rhs_B_alpha.np[ndocc_B:, :]
    # omega_jj
    rhs_B.np[ndocc_B:, :nsocc_B] = np.zeros((nsocc_B, nsocc_B))

    # NOTE::
    # ROHF::Hx expected structure
    # docc x socc | docc x virt
    # socc x socc | socc x virt
    # this translates into
    # omega_ai | omega_ar
    # -------------------
    # omega_ii | omega_ir
    #
    # and
    #
    # omega_bj | omega_bs
    # -------------------
    # omega_jj | omega_js
    # NOTE: output socc x socc (omega_ii) is always set to zero by ROHF.Hx

    print(f"{rhs_A.np.shape=}")
    print(f"{rhs_B.np.shape=}")
    print(f"{rhs_A.np=}")
    print(f"{rhs_B.np=}")
    print(f"{rhs_A_alpha.np=}")
    print(f"{rhs_A_beta.np=}")
    # call the actual solver
    t_A, t_B = _sapt_cpscf_solve(cache, jk, rhs_A, rhs_B, maxiter, conv)
    print(f"{t_A.np.shape=}")
    print(f"{t_B.np.shape=}")

    # re-pack it to alpha & beta spin-blocks and compute 20ind,resp for quick check
    # A part
    t_alpha_A = rhs_A_alpha.clone()
    t_beta_A = rhs_A_beta.clone()
    t_alpha_A.zero()
    t_beta_A.zero()
    # t_alpha = (t_ar, t_ir)
    t_ar = t_A.np[:ndocc_A, nsocc_A:].copy()
    t_ir = t_A.np[ndocc_A:, nsocc_A:].copy()
    t_ai = t_A.np[:ndocc_A, :nsocc_A].copy()

    # NOTE: correction coefficients
    # NOTE: WTF this 2 comes from
    # NOTE: H (-t) = omega
    # A
    t_ai *= -2
    t_ar *= -4  ## 4 here so results match for closed-shell
    t_ir *= -1

    # sanity checks
    assert t_ar.shape == (ndocc_A, nvirt_A)
    assert t_ir.shape == (nsocc_A, nvirt_A)
    t_alpha_A.np[:ndocc_A, :] = t_ar
    if nsocc_A:
        t_alpha_A.np[ndocc_A:, :] = t_ir
    # t_beta =  (t_ar, t_ai)

    # sanity checks
    assert t_ai.shape == (ndocc_A, nsocc_A)
    t_beta_A.np[:, :nvirt_A] = t_ar
    if nsocc_A:
        t_beta_A.np[:, nvirt_A:] = t_ai

    # B part
    t_alpha_B = rhs_B_alpha.clone()
    t_beta_B = rhs_B_beta.clone()
    t_alpha_B.zero()
    t_beta_B.zero()

    # t_alpha = (t_bs, t_js)
    t_bs = t_B.np[:ndocc_B, nsocc_B:].copy()
    t_js = t_B.np[ndocc_B:, nsocc_B:].copy()
    t_bj = t_B.np[:ndocc_B, :nsocc_B].copy()

    # NOTE: correction coefficients
    # NOTE: WTF this 2 comes from
    # NOTE: H (-t) = omega
    t_bj *= -2
    t_bs *= -4  ## 4 here so results match for closed-shell
    t_js *= -1

    # sanity checks
    assert t_bs.shape == (ndocc_B, nvirt_B)
    assert t_js.shape == (nsocc_B, nvirt_B)
    t_alpha_B.np[:ndocc_B, :] = t_bs
    if nsocc_B:
        t_alpha_B.np[ndocc_B:, :] = t_js
    # t_beta =  (t_bs, t_bj)
    assert t_bj.shape == (ndocc_B, nsocc_B)
    t_beta_B.np[:, :nvirt_B] = t_bs
    if nsocc_B:
        t_beta_B.np[:, nvirt_B:] = t_bj

    # A<-B, in spin blocks
    E20ind_resp_A_B = 0
    E20ind_resp_A_B += np.einsum("ij,ij", t_alpha_A.np, rhs_A_alpha.np)
    E20ind_resp_A_B += np.einsum("ij,ij", t_beta_A.np, rhs_A_beta.np)

    # B<-A, in spin blocks
    E20ind_resp_B_A = 0
    E20ind_resp_B_A += np.einsum("ij,ij", t_alpha_B.np, rhs_B_alpha.np)
    E20ind_resp_B_A += np.einsum("ij,ij", t_beta_B.np, rhs_B_beta.np)

    # total 20ind,resp
    E20ind_resp = E20ind_resp_A_B + E20ind_resp_B_A

    # debug print
    print(f"E20ind,resp(A<-B): {E20ind_resp_A_B}")
    print(f"E20ind,resp(B<-A): {E20ind_resp_B_A}")
    print(f"E20ind,resp      : {E20ind_resp}")

    ret_values = OrderedDict({
        "Ind20,r(A<-B)": E20ind_resp_A_B,
        "Ind20,r(A->B)": E20ind_resp_B_A,
        "Ind20,r": E20ind_resp
    })

    ret_arrays = OrderedDict({"t_ai": t_ai, "t_ar": t_ar, "t_ir": t_ir, "t_bj": t_bj, "t_bs": t_bs, "t_js": t_js})
    return ret_values, ret_arrays


def _sapt_cpscf_solve(cache, jk, rhsA, rhsB, maxiter, conv):
    """
    Solve the CP-ROHF for SAPT induction amplitudes.
    """

    cache["wfn_A"].set_jk(jk)
    cache["wfn_B"].set_jk(jk)

    # NOTE: in ROHF:Hx
    # both X and RHS
    # have sizes:
    # (docc, socc) | (docc, virt)
    # (socc, socc) | (socc, virt)
    # in turn our pre-conditioner should be
    # (eps_socc - eps_docc) | (eps_virt - eps_docc)
    # (1,)*(socc, socc)     | (eps_virt - eps_socc)
    # this way apply_denominator should not be singular (eps_socc - eps_socc) -> 0

    ndocc_A = cache["ndocc_A"]
    nsocc_A = cache["nsocc_A"]
    ndocc_B = cache["ndocc_B"]
    nsocc_B = cache["nsocc_B"]

    # Make a preconditioner function
    P_A = core.Matrix(cache["eps_docc_A"].shape[0] + cache["eps_socc_A"].shape[0],
                      cache["eps_socc_A"].shape[0] + cache["eps_vir_A"].shape[0])
    # where does this go?
    eps_ai_A = (cache["eps_docc_A"].np.reshape(-1, 1) - cache["eps_socc_A"].np)
    eps_ar_A = (cache["eps_docc_A"].np.reshape(-1, 1) - cache["eps_vir_A"].np)
    eps_ir_A = (cache["eps_socc_A"].np.reshape(-1, 1) - cache["eps_vir_A"].np)
    eps_ii_A = np.ones((cache["nsocc_A"], cache["nsocc_A"]))

    print(f"{eps_ai_A.shape=}")
    print(f"{eps_ar_A.shape=}")
    print(f"{eps_ir_A.shape=}")
    print(f"{eps_ii_A.shape=}")
    print(f"{P_A.np.shape=}")
    # ai
    P_A.np[:ndocc_A, :nsocc_A] = eps_ai_A
    # ar
    P_A.np[:ndocc_A, nsocc_A:] = eps_ar_A
    # ir
    P_A.np[ndocc_A:, nsocc_A:] = eps_ir_A
    # ii
    P_A.np[ndocc_A:, :nsocc_A] = eps_ii_A

    print(f"{cache['eps_docc_B'].shape[0]=}")
    print(f"{cache['eps_socc_B'].shape[0]=}")
    print(f"{cache['eps_vir_B'].shape[0]=}")
    print(f"{cache['ndocc_B']=}")
    print(f"{cache['nsocc_B']=}")
    print(f"{cache['nvir_B']=}")
    P_B = core.Matrix(cache["eps_docc_B"].shape[0] + cache["eps_socc_B"].shape[0],
                      cache["eps_socc_B"].shape[0] + cache["eps_vir_B"].shape[0])
    eps_ai_B = (cache["eps_docc_B"].np.reshape(-1, 1) - cache["eps_socc_B"].np)
    eps_ar_B = (cache["eps_docc_B"].np.reshape(-1, 1) - cache["eps_vir_B"].np)
    eps_ir_B = (cache["eps_socc_B"].np.reshape(-1, 1) - cache["eps_vir_B"].np)
    eps_ii_B = np.ones((nsocc_B, nsocc_B))

    print(f"{eps_ai_B.shape=}")
    print(f"{eps_ar_B.shape=}")
    print(f"{eps_ir_B.shape=}")
    print(f"{eps_ii_B.shape=}")
    print(f"{P_B.np.shape=}")
    # ai
    P_B.np[:ndocc_B, :nsocc_B] = eps_ai_B
    # ar
    P_B.np[:ndocc_B, nsocc_B:] = eps_ar_B
    # ir
    P_B.np[ndocc_B:, nsocc_B:] = eps_ir_B
    # ii
    P_B.np[ndocc_B:, :nsocc_B] = eps_ii_B

    # Preconditioner function
    def apply_precon(x_vec, act_mask):
        # NOTE: A.apply_denominator(B) does
        # element-wise A_ij = A_ij/B_ij
        if act_mask[0]:
            pA = x_vec[0].clone()
            pA.apply_denominator(P_A)
        else:
            pA = False

        if act_mask[1]:
            pB = x_vec[1].clone()
            pB.apply_denominator(P_B)
        else:
            pB = False

        # NOTE: short-circut the logic for now
        #pA, pB = (x_vec[0].clone(), x_vec[1].clone())
        return [pA, pB]

    # Hx function
    def hessian_vec(x_vec, act_mask):
        if act_mask[0]:
            xA = cache["wfn_A"].cphf_Hx([x_vec[0]])[0]
        else:
            xA = False

        if act_mask[1]:
            xB = cache["wfn_B"].cphf_Hx([x_vec[1]])[0]
        else:
            xB = False

        return [xA, xB]

    # Manipulate the printing
    sep_size = 51
    core.print_out("   " + ("-" * sep_size) + "\n")
    core.print_out("   " + "SAPT Coupled Induction (ROHF) Solver".center(sep_size) + "\n")
    core.print_out("   " + ("-" * sep_size) + "\n")
    core.print_out("    Maxiter             = %11d\n" % maxiter)
    core.print_out("    Convergence         = %11.3E\n" % conv)
    core.print_out("   " + ("-" * sep_size) + "\n")

    tstart = time.time()
    core.print_out("     %4s %12s     %12s     %9s\n" % ("Iter", "(A<-B)", "(B->A)", "Time [s]"))
    core.print_out("   " + ("-" * sep_size) + "\n")

    start_resid = [rhsA.sum_of_squares(), rhsB.sum_of_squares()]

    # print function
    def pfunc(niter, x_vec, r_vec):
        if niter == 0:
            niter = "Guess"
        else:
            niter = ("%5d" % niter)

        # Compute IndAB
        valA = (r_vec[0].sum_of_squares() / start_resid[0])**0.5
        if valA < conv:
            cA = "*"
        else:
            cA = " "

        # Compute IndBA
        valB = (r_vec[1].sum_of_squares() / start_resid[1])**0.5
        if valB < conv:
            cB = "*"
        else:
            cB = " "

        core.print_out("    %5s %15.6e%1s %15.6e%1s %9d\n" % (niter, valA, cA, valB, cB, time.time() - tstart))
        return [valA, valB]

    # Compute the solver
    vecs, resid = solvers.cg_solver([rhsA, rhsB],
                                    hessian_vec,
                                    apply_precon,
                                    guess=None,
                                    maxiter=maxiter,
                                    rcond=conv,
                                    printlvl=0,
                                    printer=pfunc)
    core.print_out("   " + ("-" * sep_size) + "\n")

    return vecs
