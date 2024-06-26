from pathlib import Path

from addons import *

import psi4


@uusing("dftd3")
@ctest_labeler("quick;sapt;cart;fsapt")
def test_fsaptd_terms():
    fsaptpy_installed = (Path(psi4.core.get_datadir()) / "fsapt" / "fsapt.py").resolve()

    ctest_runner(
        __file__,
        extra_infiles=[
            fsaptpy_installed,
        ],
    )
