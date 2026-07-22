#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SOURCE_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)

PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_EXPORT_VENV=${MODEL_EXPORT_VENV:-"${SOURCE_DIR}/.model-export-venv"}
MODEL_EXPORT_LOCK_OUTPUT=${MODEL_EXPORT_LOCK_OUTPUT:-"${SOURCE_DIR}/tools/model_export/requirements.lock"}
MODEL_EXPORT_REFRESH_LOCK=${MODEL_EXPORT_REFRESH_LOCK:-0}
REQUIREMENTS_IN="${SOURCE_DIR}/tools/model_export/requirements.in"
PIP_BOOTSTRAP_VERSION='pip==26.0.1'

require_python_312() {
  local python_path=$1
  local description=$2
  local version

  version=$("${python_path}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
  if [[ ${version} != 3.12 ]]; then
    printf 'error: Python 3.12 is required for %s (found %s)\n' "${description}" "${version}" >&2
    return 1
  fi
}

require_python_312 "${PYTHON_BIN}" 'the requested PYTHON_BIN'

if [[ ! -x "${MODEL_EXPORT_VENV}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${MODEL_EXPORT_VENV}"
fi

venv_python="${MODEL_EXPORT_VENV}/bin/python"
if ! require_python_312 "${venv_python}" "the existing model export venv at ${MODEL_EXPORT_VENV}"; then
  printf 'Remove that venv yourself or set MODEL_EXPORT_VENV to a different Python 3.12 venv path.\n' >&2
  exit 1
fi
"${venv_python}" -m pip install --upgrade "${PIP_BOOTSTRAP_VERSION}"

if [[ -s "${MODEL_EXPORT_LOCK_OUTPUT}" && ${MODEL_EXPORT_REFRESH_LOCK} != 1 ]]; then
  "${venv_python}" -m pip install -r "${MODEL_EXPORT_LOCK_OUTPUT}"
else
  lock_directory=$(dirname -- "${MODEL_EXPORT_LOCK_OUTPUT}")
  mkdir -p "${lock_directory}"
  temporary_lock=$(mktemp "${lock_directory}/.requirements.lock.XXXXXX")
  resolver_venv=$(mktemp -d "${TMPDIR:-/tmp}/rppg-model-export-resolver.XXXXXX")
  cleanup_resolver() {
    rm -f "${temporary_lock}"
    rm -rf "${resolver_venv}"
  }
  trap cleanup_resolver EXIT

  "${PYTHON_BIN}" -m venv "${resolver_venv}"
  resolver_python="${resolver_venv}/bin/python"
  require_python_312 "${resolver_python}" 'the temporary model export resolver venv'
  "${resolver_python}" -m pip install --upgrade "${PIP_BOOTSTRAP_VERSION}"
  "${resolver_python}" -m pip install -r "${REQUIREMENTS_IN}"
  "${resolver_python}" -m pip freeze | LC_ALL=C sort >"${temporary_lock}"
  chmod 0644 "${temporary_lock}"
  mv -f "${temporary_lock}" "${MODEL_EXPORT_LOCK_OUTPUT}"
  "${venv_python}" -m pip install -r "${MODEL_EXPORT_LOCK_OUTPUT}"
  trap - EXIT
  cleanup_resolver
fi

printf 'Model export environment ready: %s\n' "${venv_python}"
