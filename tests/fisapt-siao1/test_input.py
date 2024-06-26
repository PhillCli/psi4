from pathlib import Path

from addons import *

import psi4


@ctest_labeler("sapt")
def test_fisapt_siao1():
    fsaptpy_installed = (Path(psi4.core.get_datadir()) / "fsapt" / "fsapt.py").resolve()

    ctest_runner(
        __file__,
        extra_infiles=[
            fsaptpy_installed,
        ],
    )
