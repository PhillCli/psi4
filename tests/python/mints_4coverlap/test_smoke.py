import numpy as np
import pytest

import psi4


def test_4c_overlap_full():
    # Create a simple molecule
    mol = psi4.geometry("""
    H
    H 1 1.0
    """)

    basis_4 = basis_3 = basis_2 = basis_1 = psi4.core.BasisSet.build(
        mol, "BASIS", "6-31G", quiet=True
    )

    # Create MintsHelper object
    mints = psi4.core.MintsHelper(basis_1)

    # Compute 4-center overlap integrals (PQRS)
    overlap_4c_npy = mints.ao_4coverlap(basis_1, basis_2, basis_3, basis_4).np

    # Basic tests
    nbasis = basis_1.nbf()
    print(f"{nbasis=}")

    assert overlap_4c_npy.ndim == 4
    assert overlap_4c_npy.shape == (
        basis_1.nbf(),
        basis_2.nbf(),
        basis_3.nbf(),
        basis_4.nbf(),
    )

    # Test permutational symmetry
    for i in range(nbasis):
        for j in range(nbasis):
            for k in range(nbasis):
                for l in range(nbasis):
                    assert (
                        abs(overlap_4c_npy[i, j, k, l] - overlap_4c_npy[j, i, k, l])
                        < 1e-10
                    )
                    assert (
                        abs(overlap_4c_npy[i, j, k, l] - overlap_4c_npy[i, j, l, k])
                        < 1e-10
                    )


def test_4c_overlap_diag():
    # Create a simple molecule
    mol = psi4.geometry("""
    H
    H 1 1.0
    """)

    basis_2 = basis_1 = psi4.core.BasisSet.build(mol, "BASIS", "cc-pvdz", quiet=True)

    # Create MintsHelper object
    mints = psi4.core.MintsHelper(basis_1)

    # Compute 4-center overlap integrals (PQPQ)
    overlap_4c_diag_npy = mints.ao_4coverlap_diag(basis_1, basis_2).np
    overlap_4c_full_npy = mints.ao_4coverlap(basis_1, basis_2, basis_1, basis_2).np

    # Basic tests
    nbasis = basis_1.nbf()
    print(f"{nbasis=}")

    assert overlap_4c_diag_npy.ndim == 2
    assert overlap_4c_diag_npy.shape == (
        basis_1.nbf(),
        basis_2.nbf(),
    )
    # print("matrix from diagonal call")
    # print(f"\n{overlap_4c_diag_npy}")
    # print("matrix from full call")
    # print(f"\n{overlap_4c_full_npy.reshape(nbasis * nbasis, nbasis * nbasis)}")

    # Test permutational symmetry
    for i in range(nbasis):
        for j in range(nbasis):
            assert abs(overlap_4c_diag_npy[i, j] - overlap_4c_diag_npy[j, i]) < 1e-10
            assert abs(overlap_4c_diag_npy[i, j] - overlap_4c_diag_npy[i, j]) < 1e-10

            assert (
                abs(overlap_4c_diag_npy[i, j] - overlap_4c_full_npy[i, j, i, j]) < 1e-10
            ), "Diagonal elements from .ao_4coverlap_diag() should be the same as the ones from full 4c overlap elements!"
