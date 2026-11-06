#!/usr/bin/env bash
# Build the PDF and an arXiv source archive from paper/preprint/.
# Usage (from the repository root):  bash paper/preprint/make_arxiv.sh
# Requires: Python with the repository dependencies, pdflatex, bibtex, latexmk.
set -euo pipefail
cd "$(dirname "$0")"
python ../../scripts/build_preprint_assets.py > /dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1
if grep -E "Warning|Overfull|undefined" main.log; then
  echo "LaTeX warnings found (see main.log)"; exit 1
fi
STAGE=$(mktemp -d)
mkdir -p "$STAGE/generated" "$STAGE/figures"
cp main.tex appendix.tex references.bib main.bbl "$STAGE/"
cp generated/*.tex "$STAGE/generated/"
cp figures/*.pdf "$STAGE/figures/"
tar -czf arxiv-source.tar.gz -C "$STAGE" .
# Check that the archive builds on its own.
CHECK=$(mktemp -d)
tar -xzf arxiv-source.tar.gz -C "$CHECK"
(cd "$CHECK" && pdflatex -interaction=nonstopmode main.tex > /dev/null && pdflatex -interaction=nonstopmode main.tex > /dev/null)
test -s "$CHECK/main.pdf"
echo "arxiv-source.tar.gz builds; contents:"
tar -tzf arxiv-source.tar.gz
