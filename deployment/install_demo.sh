#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${repo_dir}/.conda-env"

if [[ ! -x "${env_dir}/bin/python" ]]; then
    conda create -y -p "${env_dir}" python=3.10 pip
fi

"${env_dir}/bin/python" -m pip install -r "${repo_dir}/deployment/requirements-demo.txt"
"${env_dir}/bin/python" -m pip install --no-deps -e "${repo_dir}/third_party/robosuite-1.4" -e "${repo_dir}"
"${env_dir}/bin/python" -m pip install 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
"${env_dir}/bin/python" -m pip check
