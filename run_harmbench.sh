#!/usr/bin/env bash
set -euo pipefail

if [[ $# == 1 && "$1" == --setup-and-train && "$(uname -s)" == Linux ]]; then
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    env_dir="/workspace/harmbench-env"

    if [[ ! -x "$env_dir/bin/python" ]]; then
        printf 'Creating isolated Python 3.11 environment...\n'
        export UV_PYTHON_INSTALL_DIR="/workspace/harmbench-python"
        uv_command=(python -m uv)
        if ! python -m uv --version >/dev/null 2>&1; then
            tools_dir="/workspace/harmbench-tools"
            python -m pip install --only-binary=:all: --target "$tools_dir" uv
            uv_command=(env "PYTHONPATH=$tools_dir" python -m uv)
        fi
        "${uv_command[@]}" python install 3.11
        "${uv_command[@]}" venv --python 3.11 --seed --allow-existing "$env_dir"
    fi

    if [[ ! -f "$env_dir/.safety-decoding-ready-v1" ]]; then
        "$env_dir/bin/python" -m pip install --only-binary=:all: torch==2.6.0 \
            --index-url https://download.pytorch.org/whl/cu124
        "$env_dir/bin/python" -m pip install \
            numpy einops omegaconf==2.3.0 \
            transformers==4.48.3 tokenizers==0.21.4 \
            'huggingface-hub>=0.24,<1' wandb tqdm sentencepiece protobuf matplotlib
        "$env_dir/bin/python" - <<'PY'
import subprocess
import sys
import torch

if sys.version_info[:2] != (3, 11) or torch.__version__ != "2.6.0+cu124":
    raise RuntimeError("Expected Python 3.11 and PyTorch 2.6.0+cu124")
abi = str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
wheel = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    f"flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abi{abi}-cp311-cp311-linux_x86_64.whl"
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "--no-deps", "--only-binary=:all:", wheel],
    check=True,
)
PY
    fi

    "$env_dir/bin/python" - <<'PY'
import sys
import torch
import flash_attn
import numpy
import transformers
import omegaconf
import wandb
import sentencepiece
from gfn import PosteriorPriorGFN
from load_model import load_model

if sys.version_info[:2] != (3, 11) or torch.__version__ != "2.6.0+cu124":
    raise RuntimeError("Unexpected environment version; inspect /workspace/harmbench-env")
if flash_attn.__version__ != "2.7.4.post1":
    raise RuntimeError("Expected FlashAttention 2.7.4.post1")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available")
if torch.cuda.get_device_capability()[0] < 8:
    raise RuntimeError("Ampere-or-newer GPU required")
q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
output = flash_attn.flash_attn_func(q, q, q)
output.sum().backward()
torch.cuda.synchronize()
if not torch.isfinite(output).all() or not torch.isfinite(q.grad).all():
    raise RuntimeError("FlashAttention CUDA forward/backward check failed")
print("Environment ready:", sys.executable, torch.__version__, torch.cuda.get_device_name(0), flush=True)
PY
    touch "$env_dir/.safety-decoding-ready-v1"
    "$env_dir/bin/python" -m unittest discover -s tests -p 'test_*.py'
    exec "$env_dir/bin/python" -u finetune_cond_data_harmbench.py
fi

if (( $# != 0 )); then
    printf 'This launcher accepts no overrides. Edit configs/ft_harmbench.yaml instead.\n' >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

if [[ "$(uname -s)" == Darwin ]]; then
    pod="root@103.207.149.171"
    port="14309"
    key="$HOME/.ssh/id_ed25519"

    if [[ ! -f "$key" ]]; then
        printf 'SSH key not found: %s\n' "$key" >&2
        exit 1
    fi

    ssh -i "$key" -p "$port" "$pod" 'mkdir -p /workspace'
    scp -i "$key" -P "$port" -r "$PWD" "$pod:/workspace/"
    exec ssh -t -i "$key" -p "$port" "$pod" \
        'cd /workspace/safety-decoding && bash run_harmbench.sh'
fi

if ! command -v tmux >/dev/null 2>&1; then
    printf 'tmux is required. Install it on the pod before launching.\n' >&2
    exit 1
fi

if [[ ! -t 0 || ! -t 1 ]]; then
    printf 'An interactive terminal is required. When launching via SSH, use ssh -t.\n' >&2
    exit 1
fi

if ! tmux has-session -t '=safety-decoding' 2>/dev/null; then
    tmux new-session -d -s safety-decoding -c "$PWD" \
        'exec bash run_harmbench.sh --setup-and-train' \; \
        set-option -w -t safety-decoding remain-on-exit on
elif [[ "$(tmux display-message -p -t '=safety-decoding' '#{pane_dead}')" == 1 ]]; then
    tmux respawn-pane -t '=safety-decoding' -c "$PWD" \
        'exec bash run_harmbench.sh --setup-and-train'
fi

printf 'Detach with Ctrl+B, then D. The pane retains output after training exits.\n'
if [[ -n "${TMUX:-}" ]]; then
    exec tmux switch-client -t safety-decoding
fi
exec tmux attach-session -t safety-decoding
