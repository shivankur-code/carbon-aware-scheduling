#!/usr/bin/env bash
# Prepare a fresh GPU machine (Ubuntu + NVIDIA driver) for gpu_bench.
set -euo pipefail
nvidia-smi --query-gpu=index,name,driver_version,power.default_limit,power.min_limit,power.max_limit,memory.total --format=csv
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "vllm" "nvidia-ml-py" aiohttp pyyaml
pip install -e ".[gpu]"
python - <<'PY'
import vllm; print("vllm", vllm.__version__)
PY
# Power capping needs root. Check it now, not three hours into a sweep.
LIMIT=$(nvidia-smi -i 0 --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
if nvidia-smi -i 0 -pl "$LIMIT" >/dev/null 2>&1 || sudo -n nvidia-smi -i 0 -pl "$LIMIT" >/dev/null 2>&1; then
  echo "OK: power capping works"
else
  echo "WARNING: cannot set power limits (no root). Use a VM / bare-metal instance, not a shared container."
fi
