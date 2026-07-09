#!/usr/bin/env bash
# Run the real Evo 2 7B pipeline (with gene boundaries) on each locus in
# ~/SerTnpB-Evo.fasta. Each locus is named by its accession; outputs go to ~/runs/<name>/.
cd ~/DNA-Entropy || exit 1
mkdir -p ~/loci ~/runs
awk '/^>/{n++} {f=sprintf("%s/loci/locus%d.fa", ENVIRON["HOME"], n); print > f}' ~/SerTnpB-Evo.fasta
echo "===== EVO RUN START $(date) ====="
for f in ~/loci/locus*.fa; do
  acc=$(head -1 "$f" | sed 's/^>//' | awk '{print $1}')
  echo "----- $f  -> name=$acc -----"
  python3 -m dna_entropy.cli run -i "$f" --name "$acc" --predictor evo --genes --out ~/runs 2>&1 | tail -10
done
echo "===== EVO RUN DONE $(date) ====="
ls -R ~/runs
