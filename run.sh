#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src:$PYTHONPATH"
exec "${PY:-python3}" -m vrgr.cli "$@"
