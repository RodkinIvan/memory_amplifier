#!/bin/bash
# Run by cluster/route.py inside the new env right after `conda env create -f env.yml` ($PY = the env's python, $CONDA_PREFIX = the env).
# flash-attn is not in env.yml: it needs torch at build time and --no-build-isolation; pip takes the prebuilt wheel from its GitHub releases when one matches
# (torch 2.6, cu12, python 3.11) and compiles with nvcc otherwise.
set -e
"$PY" -m pip install --no-build-isolation flash-attn==2.7.4.post1
"$PY" -c "import torch, fla, flash_attn; print('env ok:', torch.__version__, fla.__version__, flash_attn.__version__)"
