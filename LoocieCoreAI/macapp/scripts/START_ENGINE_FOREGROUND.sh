#!/bin/zsh
set -euo pipefail

source "/Volumes/LoocieCoreAI/BuildCache/_Python/loocie-v2-venv/bin/activate"
cd "/Volumes/LoocieCoreAI/LoocieCoreAI_Core/LoocieAI_V2_Master/LoocieCoreAI/engine"

exec python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
