#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_EXPORT_VENV=${MODEL_EXPORT_VENV:-"${SOURCE_DIR}/.model-export-venv"}
MODEL_EXPORT_LOCK_OUTPUT=${MODEL_EXPORT_LOCK_OUTPUT:-"${SOURCE_DIR}/tools/model_export/requirements.lock"}
MODEL_EXPORT_REFRESH_LOCK=${MODEL_EXPORT_REFRESH_LOCK:-0}
REQUIREMENTS_IN="${SOURCE_DIR}/tools/model_export/requirements.in"

python_version=$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ ${python_version} != 3.12 ]]; then
  printf 'error: Python 3.12 is required for the model export environment (found %s)\n' \
    "${python_version}" >&2
  exit 1
fi

if [[ ! -x "${MODEL_EXPORT_VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${MODEL_EXPORT_VENV}"
fi

venv_python="${MODEL_EXPORT_VENV}/bin/python"
"${venv_python}" -m pip install --upgrade pip

if [[ -s "${MODEL_EXPORT_LOCK_OUTPUT}" && ${MODEL_EXPORT_REFRESH_LOCK} != 1 ]]; then
  "${venv_python}" -m pip install -r "${MODEL_EXPORT_LOCK_OUTPUT}"
else
  "${venv_python}" -m pip install -r "${REQUIREMENTS_IN}"

  lock_directory=$(dirname -- "${MODEL_EXPORT_LOCK_OUTPUT}")
  mkdir -p "${lock_directory}"
  temporary_lock=$(mktemp "${lock_directory}/.requirements.lock.XXXXXX")
  trap 'rm -f "${temporary_lock}"' EXIT
  "${venv_python}" -m pip freeze | LC_ALL=C sort >"${temporary_lock}"
  mv -f "${temporary_lock}" "${MODEL_EXPORT_LOCK_OUTPUT}"
  trap - EXIT
fi

printf 'Model export environment ready: %s\n' "${venv_python}"
