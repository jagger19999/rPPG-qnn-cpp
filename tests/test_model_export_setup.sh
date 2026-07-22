#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory is required}
SETUP_SCRIPT="${SOURCE_DIR}/scripts/setup_model_export_macos.sh"

fail() {
  printf 'test_model_export_setup: %s\n' "$1" >&2
  exit 1
}

expect_failure() {
  local output_file=$1
  shift
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
}

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/rppg-model-export.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT

fake_python="${tmp_dir}/fake-python"
cat >"${fake_python}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

interpreter_version=${FAKE_INTERPRETER_VERSION:-${FAKE_PYTHON_VERSION:-3.12}}
invocation_name=${FAKE_PYTHON_INVOCATION_NAME:-$0}

if [[ ${1:-} == -c ]]; then
  printf '%s\n' "${interpreter_version}"
  exit 0
fi

if [[ ${1:-} == -m && ${2:-} == venv ]]; then
  venv_dir=${3:?venv directory is required}
  printf 'venv <%s>\n' "${venv_dir}" >>"${FAKE_PIP_LOG:?}"
  mkdir -p "${venv_dir}/bin"
  printf '#!/usr/bin/env bash\nFAKE_PYTHON_INVOCATION_NAME="$0" FAKE_INTERPRETER_VERSION="%s" exec "%s" "$@"\n' \
    "${FAKE_VENV_VERSION:-3.12}" "${FAKE_PYTHON_PATH:?}" >"${venv_dir}/bin/python"
  chmod +x "${venv_dir}/bin/python"
  exit 0
fi

if [[ ${1:-} == -m && ${2:-} == pip ]]; then
  printf 'pip <%s> %s\n' "${invocation_name}" "$*" >>"${FAKE_PIP_LOG:?}"
  if [[ ${3:-} == freeze ]]; then
    if [[ ${FAKE_FAIL_FREEZE:-0} == 1 ]]; then
      exit 9
    fi
    printf '%s\n' 'numpy==2.2.0' 'torch==2.12.1'
  fi
  exit 0
fi

printf 'unexpected fake Python invocation: %s\n' "$*" >&2
exit 1
EOF
chmod +x "${fake_python}"

run_setup() {
  local venv_dir=$1
  local lock_output=$2
  local pip_log=$3
  local refresh=$4
  shift 4
  FAKE_PYTHON_PATH="${fake_python}" FAKE_PIP_LOG="${pip_log}" PYTHON_BIN="${fake_python}" \
    MODEL_EXPORT_VENV="${venv_dir}" MODEL_EXPORT_LOCK_OUTPUT="${lock_output}" \
    MODEL_EXPORT_REFRESH_LOCK="${refresh}" "$@" "${SETUP_SCRIPT}"
}

venv_dir="${tmp_dir}/venv with spaces"
lock_output="${tmp_dir}/requirements.lock"
pip_log="${tmp_dir}/refresh.log"
run_setup "${venv_dir}" "${lock_output}" "${pip_log}" 1

[[ -x "${venv_dir}/bin/python" ]] || fail 'setup did not create an executable work venv Python'
grep -Fqx 'torch==2.12.1' "${lock_output}" || fail 'refresh did not write torch==2.12.1 to the lock'
grep -Fqx "pip <${venv_dir}/bin/python> -m pip install --upgrade pip==26.0.1" "${pip_log}" ||
  fail 'work venv did not use the pinned pip bootstrap'
grep -Fqx "pip <${venv_dir}/bin/python> -m pip install -r ${lock_output}" "${pip_log}" ||
  fail 'work venv did not install the generated lock exactly'
resolver_dir=$(sed -n 's/^venv <\(.*\)>$/\1/p' "${pip_log}" | sed -n '2p')
[[ -n ${resolver_dir} && ${resolver_dir} != "${venv_dir}" ]] || fail 'refresh did not create an independent resolver venv'
grep -Fqx "pip <${resolver_dir}/bin/python> -m pip freeze" "${pip_log}" ||
  fail 'refresh did not freeze from the resolver venv'
! grep -Fqx "pip <${venv_dir}/bin/python> -m pip freeze" "${pip_log}" ||
  fail 'refresh froze the work venv'
[[ ! -e "${resolver_dir}" ]] || fail 'refresh did not remove the resolver venv'

existing_log="${tmp_dir}/existing-lock.log"
: >"${existing_log}"
run_setup "${venv_dir}" "${lock_output}" "${existing_log}" 0
run_setup "${venv_dir}" "${lock_output}" "${existing_log}" 0
[[ $(grep -Fxc "pip <${venv_dir}/bin/python> -m pip install -r ${lock_output}" "${existing_log}") -eq 2 ]] ||
  fail 'existing lock was not installed exactly once per repeated run'
[[ $(grep -c '^venv <' "${existing_log}" || true) -eq 0 ]] ||
  fail 'repeated runs unexpectedly recreated a venv'
! grep -Fq "requirements.in" "${existing_log}" || fail 'existing lock branch installed requirements.in'

wrong_venv="${tmp_dir}/wrong-version-venv"
FAKE_PYTHON_PATH="${fake_python}" FAKE_PIP_LOG="${tmp_dir}/wrong-venv-create.log" \
  FAKE_VENV_VERSION=3.14 "${fake_python}" -m venv "${wrong_venv}"
wrong_version_log="${tmp_dir}/wrong-version.log"
expect_failure "${wrong_version_log}" env FAKE_PYTHON_PATH="${fake_python}" \
  FAKE_PIP_LOG="${tmp_dir}/wrong-venv.log" PYTHON_BIN="${fake_python}" \
  MODEL_EXPORT_VENV="${wrong_venv}" MODEL_EXPORT_LOCK_OUTPUT="${tmp_dir}/wrong-version.lock" \
  "${SETUP_SCRIPT}"
grep -Fq 'Python 3.12' "${wrong_version_log}" || fail 'wrong venv version did not require Python 3.12'
grep -Eq 'delete|remove|MODEL_EXPORT_VENV' "${wrong_version_log}" ||
  fail 'wrong venv version did not explain how to choose a replacement'

old_lock="${tmp_dir}/old.lock"
printf 'old-lock-content\n' >"${old_lock}"
failed_refresh_log="${tmp_dir}/failed-refresh.log"
expect_failure "${tmp_dir}/freeze-failure.log" env FAKE_PYTHON_PATH="${fake_python}" \
  FAKE_PIP_LOG="${failed_refresh_log}" FAKE_FAIL_FREEZE=1 PYTHON_BIN="${fake_python}" \
  MODEL_EXPORT_VENV="${tmp_dir}/failure-work-venv" MODEL_EXPORT_LOCK_OUTPUT="${old_lock}" \
  MODEL_EXPORT_REFRESH_LOCK=1 "${SETUP_SCRIPT}"
[[ $(<"${old_lock}") == old-lock-content ]] || fail 'freeze failure changed the existing lock'
failed_resolver=$(sed -n 's/^venv <\(.*\)>$/\1/p' "${failed_refresh_log}" | sed -n '2p')
[[ -n ${failed_resolver} && ! -e "${failed_resolver}" ]] ||
  fail 'freeze failure did not remove the resolver venv'

wrong_python_log="${tmp_dir}/wrong-python.log"
expect_failure "${wrong_python_log}" env FAKE_PYTHON_VERSION=3.14 FAKE_PYTHON_PATH="${fake_python}" \
  FAKE_PIP_LOG="${tmp_dir}/wrong-python-pip.log" PYTHON_BIN="${fake_python}" \
  MODEL_EXPORT_VENV="${tmp_dir}/wrong-python-venv" MODEL_EXPORT_LOCK_OUTPUT="${tmp_dir}/wrong-python.lock" \
  "${SETUP_SCRIPT}"
grep -Fq 'Python 3.12' "${wrong_python_log}" || fail 'wrong requested Python did not require Python 3.12'

printf 'test_model_export_setup: PASS\n'
