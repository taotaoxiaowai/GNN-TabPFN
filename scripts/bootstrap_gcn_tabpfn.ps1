$ErrorActionPreference = "Stop"
param(
    [string]$EnvName = "gcn_tabpfn_py311"
)

Write-Host "[1/4] Check Python version in conda env: $EnvName"
$pyVersion = conda run -n $EnvName python -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
Write-Host "Python version: $pyVersion"

if (-not $pyVersion.StartsWith("3.11.")) {
    Write-Host "Current env $EnvName is not Python 3.11."
    Write-Host "Please create and activate the 3.11 env first:"
    Write-Host "  conda env create -f environment.gcn_tabpfn-py311.yml"
    Write-Host "  conda activate gcn_tabpfn_py311"
    exit 1
}

Write-Host "[2/4] Upgrade pip"
conda run -n $EnvName python -m pip install --upgrade pip

Write-Host "[3/4] Install project package and dev dependencies"
conda run -n $EnvName python -m pip install -e .[dev]

Write-Host "[4/4] Run smoke test"
conda run -n $EnvName python -m pytest -q

Write-Host "Bootstrap completed."
