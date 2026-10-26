#!/bin/sh
# pip-installed CUDA libs live under site-packages/nvidia/*/lib; TensorFlow needs them on the loader path.
_SP="$(python -c 'import site;print(site.getsitepackages()[0])')"
export _BRAINSEG_OLD_LD="${LD_LIBRARY_PATH:-}"
for d in "$_SP"/nvidia/*/lib; do LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"; done
export LD_LIBRARY_PATH
