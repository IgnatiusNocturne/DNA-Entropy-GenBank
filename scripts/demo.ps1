# DNA-Entropy demo (Windows / mock predictor — no GPU needed).
# Run from the repo root:  .\scripts\demo.ps1
$ErrorActionPreference = "Stop"

$cli = ".\.venv\Scripts\dna-entropy.exe"

Write-Host "== DNA-Entropy demo (mock predictor) ==" -ForegroundColor Cyan
& $cli run --input tests\data\sample.fasta --name sample_locus --out out

Write-Host ""
Write-Host "== With gene boundaries (prokaryotic ORF demo) ==" -ForegroundColor Cyan
& $cli run --input tests\data\prokaryotic_demo.fasta --name prok_demo --genes --out out

Write-Host ""
Write-Host "Done. To view in IGV:" -ForegroundColor Cyan
Write-Host "  1) Genomes -> Load Genome from File -> out\sample_locus.fasta"
Write-Host "  2) File    -> Load from File        -> out\sample_locus.entropy.bedgraph"
Write-Host ""
Write-Host "For the REAL model on a GPU box, see docs/EVO_SETUP.md, then:" -ForegroundColor Cyan
Write-Host "  dna-entropy run -i locus.txt --predictor evo --model evo2_7b --genes"
