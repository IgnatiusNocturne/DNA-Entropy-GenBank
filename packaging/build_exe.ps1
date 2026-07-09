# Build the Windows .exe files.  Run from the repo root:  .\packaging\build_exe.ps1
#   dist\dna-entropy.exe  - the client (double-click wizard + full CLI)
#   dist\keep-gpu.exe     - the always-on GPU keeper (double-click, leave open)
# Both are lightweight clients: torch/evo2/flash-attn run in the cloud, not here.
# Biopython IS bundled (GenBank/FASTA I/O runs locally); it is imported lazily, so we must
# collect its submodules explicitly or PyInstaller misses them.
$ErrorActionPreference = "Stop"

.\.venv\Scripts\python.exe -m pip install --quiet pyinstaller

$common = @(
    "--onefile", "--console", "--noconfirm",
    "--exclude-module", "torch", "--exclude-module", "evo2",
    "--exclude-module", "flash_attn", "--exclude-module", "pyrodigal",
    "--collect-submodules", "Bio",
    "--add-data", "src/dna_entropy;_pkgsrc"
)

# 1. The client.
.\.venv\Scripts\python.exe -m PyInstaller @common --name dna-entropy packaging\launcher.py

# 2. The always-on GPU keeper (bundles the package source so it can upload it to the VM).
.\.venv\Scripts\python.exe -m PyInstaller @common --name keep-gpu keep_gpu.py

Write-Host ""
Write-Host "Built: dist\dna-entropy.exe  and  dist\keep-gpu.exe" -ForegroundColor Green
Write-Host "Distribute via a GitHub Release." -ForegroundColor Cyan
