#!/usr/bin/env bash
set -euo pipefail

kaggle kernels push -p kaggle/v2_s0 --accelerator NvidiaTeslaT4 --timeout 3600
