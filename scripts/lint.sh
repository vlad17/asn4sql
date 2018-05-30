#! /usr/bin/env bash

# Lints code:
#
#   # Lint asn4sql by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh asn4sql/somefile/*.py

set -euo pipefail

lint() {
    pylint "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint asn4sql
    else
        lint "$@"
    fi
}

main "$@"
