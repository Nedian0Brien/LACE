#!/usr/bin/env bash
set -euo pipefail

kaggle kernels push -p kaggle/phase3a --accelerator NvidiaTeslaT4 --timeout 3600
