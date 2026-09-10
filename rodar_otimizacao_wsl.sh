#!/usr/bin/env bash
set -euo pipefail

# Execute este arquivo DENTRO do WSL, a partir desta pasta.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$ROOT/trials"
exec python3 "$ROOT/controlador_otimizacao.py" --root "$ROOT"

