#!/usr/bin/env bash
# One-time: populate a project-local texmf tree with the font packages that the
# AAAI 2027 style (aaai2027.sty) needs but the container's minimal TeX Live is
# missing: newtx (newtxtext/newtxmath) and its mweights dependency.
#
# Run INSIDE the container so mktexlsr/updmap-user match the container's TeX:
#   apptainer exec --overlay /scratch/ab9738/stamp/overlay-25GB-500K.ext3:ro \
#     /scratch/ab9738/stamp/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
#     bash /scratch/ab9738/stamp/.vscode/texmf-setup.sh
#
# The tree itself is git-ignored; this script regenerates it. It prefers copying
# the already-populated sibling tree at dsrc/.vscode/texmf (this cluster has no
# outbound network from the container); if that is absent it falls back to the
# frozen TeX Live 2022 archive. The build recipe in settings.json exports the same
# TEXMFHOME/TEXMFVAR/TEXMFCONFIG and prepends this tree to TEXMFDBS.
set -euo pipefail
REPO=/scratch/ab9738/stamp
SIBLING=/scratch/ab9738/dsrc/.vscode/texmf
export TEXMFHOME=$REPO/.vscode/texmf
export TEXMFVAR=$REPO/.vscode/texmf-var
export TEXMFCONFIG=$REPO/.vscode/texmf-config
rm -rf "$TEXMFHOME" "$TEXMFVAR" "$TEXMFCONFIG"
mkdir -p "$TEXMFHOME" "$TEXMFVAR" "$TEXMFCONFIG"

if [ -d "$SIBLING/tex/latex/newtx" ]; then
  echo ">> copying newtx/mweights font tree from sibling $SIBLING (network-free)"
  cp -a "$SIBLING/." "$TEXMFHOME"/
else
  arch=https://ftp.math.utah.edu/pub/tex/historic/systems/texlive/2022/tlnet-final/archive
  work=$(mktemp -d)
  for pkg in newtx mweights; do
    echo ">> $pkg (TL2022 archive fallback)"
    curl -fLsS "$arch/$pkg.tar.xz" -o "$work/$pkg.tar.xz"
    rm -rf "$work/x" && mkdir -p "$work/x"
    tar xf "$work/$pkg.tar.xz" -C "$work/x"
    for d in tex fonts; do [ -d "$work/x/$d" ] && cp -a "$work/x/$d" "$TEXMFHOME"/; done
  done
  rm -rf "$work"
fi

# Filename database for the tree (build recipe adds it to TEXMFDBS).
mktexlsr "$TEXMFHOME" >/dev/null 2>&1 || true

# Enable the font maps the packages shipped, into the writable user var tree.
shopt -s nullglob
for m in "$TEXMFHOME"/fonts/map/*/*/*.map; do
  updmap-user --enable Map="$(basename "$m")" >/dev/null 2>&1 || true
done
updmap-user >/dev/null 2>&1 || true

echo "OK -- trees under $REPO/.vscode/  (texmf, texmf-var, texmf-config)"
kpsewhich newtxtext.sty newtxmath.sty mweights.sty || true
