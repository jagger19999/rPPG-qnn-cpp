#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory is required}
SETUP_SCRIPT="${SOURCE_DIR}/scripts/setup_model_export_macos.sh"

fail() {
  printf 'test_model_export_setup: %s\n' "$1" >&2
  exit 1
}

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/rppg-model-export.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT

fake_python="${tmp_dir}/fake-python"
cat >"${fake_python}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == -c ]]; then
  printf '%s\n' "${FAKE_PYTHON_VERSION:-3.12}"
  exit 0
fi

if [[ ${1:-} == -m && ${2:-} == venv ]]; then
  venv_dir=${3:?venv directory is required}
  mkdir -p "${venv_dir}/bin"
  printf '#!/usr/bin/env bash\nexec "%s" "$@"\n' "$0" >"${venv_dir}/bin/python"
  chmod +x "${venv_dir}/bin/python"
  exit 0
fi

if [[ ${1:-} == -m && ${2:-} == pip ]]; then
  printf '%s\n' "$*" >>"${FAKE_PIP_LOG:?}"
  if [[ ${3:-} == freeze ]]; then
    printf '%s\n' 'numpy==2.2.0' 'torch==2.12.1'
  fi
  exit 0
fi

printf 'unexpected fake Python invocation: %s\n' "$*" >&2
exit 1
EOF
chmod +x "${fake_python}"

venv_dir="${tmp_dir}/venv with spaces"
lock_output="${tmp_dir}/requirements.lock"
pip_log="${tmp_dir}/pip.log"
FAKE_PIP_LOG="${pip_log}" PYTHON_BIN="${fake_python}" \
  MODEL_EXPORT_VENV="${venv_dir}" MODEL_EXPORT_LOCK_OUTPUT="${lock_output}" \
  MODEL_EXPORT_REFRESH_LOCK=1 "${SETUP_SCRIPT}"

[[ -x "${venv_dir}/bin/python" ]] || fail 'setup did not create an executable venv Python'
grep -Fqx -- '-m pip install --upgrade pip' "${pip_log}" || fail 'setup did not upgrade pip'
grep -Fq -- '-m pip install -r ' "${pip_log}" || fail 'setup did not install requirements'
grep -Fqx 'torch==2.12.1' "${lock_output}" || fail 'refresh did not write torch==2.12.1 to the lock'

wrong_version_log="${tmp_dir}/wrong-version.log"
if FAKE_PYTHON_VERSION=3.14 FAKE_PIP_LOG="${pip_log}" PYTHON_BIN="${fake_python}" \
  MODEL_EXPORT_VENV="${tmp_dir}/wrong-version-venv" \
  MODEL_EXPORT_LOCK_OUTPUT="${tmp_dir}/wrong-version.lock" \
  "${SETUP_SCRIPT}" >"${wrong_version_log}" 2>&1; then
  fail 'setup unexpectedly accepted Python 3.14'
fi
grep -Fq 'Python 3.12' "${wrong_version_log}" || fail 'wrong Python version did not require Python 3.12'

printf 'test_model_export_setup: PASS\n'
