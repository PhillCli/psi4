# Windows 10 Testing Notes for psi4-path-advisor.py

1. **Install Miniconda/Miniforge** - Download from conda-forge and install to a path without spaces (e.g., `C:\miniforge3`)

2. **Open Miniforge Prompt** - Use the "Miniforge Prompt" from Start Menu (not regular cmd/PowerShell) to get conda in PATH

3. **Clone psi4 repo** - `git clone https://github.com/psi4/psi4.git && cd psi4`

4. **Test basic help** - `python conda\psi4-path-advisor.py --help`

5. **Test env generation** - `python conda\psi4-path-advisor.py env -n win_test --disable addons docs test`

6. **Verify output uses `conda.bat`** - The command should show `conda.bat env create ...` (not just `conda`)

7. **Test with only mamba installed** - Uninstall conda, install mamba standalone, verify it falls back to `mamba.bat`

8. **Test with only micromamba** - Install micromamba standalone, verify it uses `micromamba.exe` and `create` (not `env create`)

9. **Test cmake subcommand** - `python conda\psi4-path-advisor.py cmake` - verify cache file generation with Windows paths

10. **Key validation** - Confirm `get_conda_exe()` returns `None` (not `micromamba`) when no package manager is in PATH, and that the `--offline-conda` flag still works in that scenario
