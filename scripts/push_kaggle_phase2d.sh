#!/usr/bin/env bash
set -euo pipefail

kaggle kernels push -p kaggle/phase2d --accelerator NvidiaTeslaT4 --timeout 3600
