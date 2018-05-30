#! /usr/bin/env bash
# ./scripts/format.sh
# Auto-formats the repository according to pep8 with yapf in parallel.

set -euo pipefail

yapf --in-place --recursive --parallel asn4sql
