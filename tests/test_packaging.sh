#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory is required}
BUILD_SCRIPT="${SOURCE_DIR}/scripts/build_linux.sh"
SMOKE_SCRIPT="${SOURCE_DIR}/scripts/run_smoke.sh"
LAUNCHER="${SOURCE_DIR}/packaging/run_rppg_qnn.sh"
DEFAULT_CONFIG="${SOURCE_DIR}/config/runtime-defaults.env"

fail() {
  printf 'test_packaging: %s\n' "$1" >&2
  exit 1
}

expect_failure() {
  local output_file=$1
  shift
  if "$@" >"${output_file}" 2>&1; then
    fail "command unexpectedly succeeded: $*"
  fi
}

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/rppg-qnn-packaging.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT

expect_failure "${tmp_dir}/unknown.log" "${BUILD_SCRIPT}" unsupported
grep -q 'expected native or aarch64' "${tmp_dir}/unknown.log" ||
  fail 'unknown build mode did not explain the accepted modes'

expect_failure "${tmp_dir}/prefix.log" env -u AARCH64_TOOLCHAIN_PREFIX \
  AARCH64_SYSROOT=/tmp/sysroot "${BUILD_SCRIPT}" aarch64
grep -q 'AARCH64_TOOLCHAIN_PREFIX' "${tmp_dir}/prefix.log" ||
  fail 'aarch64 build did not require AARCH64_TOOLCHAIN_PREFIX'

expect_failure "${tmp_dir}/sysroot.log" env -u AARCH64_SYSROOT \
  AARCH64_TOOLCHAIN_PREFIX=aarch64-linux-gnu- "${BUILD_SCRIPT}" aarch64
grep -q 'AARCH64_SYSROOT' "${tmp_dir}/sysroot.log" ||
  fail 'aarch64 build did not require AARCH64_SYSROOT'

package_dir="${tmp_dir}/package"
mkdir -p "${package_dir}/bin" "${package_dir}/lib"
package_dir=$(CDPATH= cd -- "${package_dir}" && pwd)
cp "${LAUNCHER}" "${package_dir}/bin/run_rppg_qnn.sh"
chmod +x "${package_dir}/bin/run_rppg_qnn.sh"
cat >"${package_dir}/bin/rppg_qnn_live" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${LD_LIBRARY_PATH-}"
printf '<%s>\n' "$@"
EOF
chmod +x "${package_dir}/bin/rppg_qnn_live"

launcher_output=$(QAIRT_TARGET_LIB_DIR=/qairt/target LD_LIBRARY_PATH=/existing/lib \
  "${package_dir}/bin/run_rppg_qnn.sh" 'two words' --flag)
expected_path="${package_dir}/lib:/qairt/target:/existing/lib"
[[ "${launcher_output}" == "${expected_path}"$'\n''<two words>'$'\n''<--flag>' ]] ||
  fail 'launcher did not preserve the controlled library path and arguments'

launcher_output=$(env -u QAIRT_TARGET_LIB_DIR LD_LIBRARY_PATH=/existing/lib \
  "${package_dir}/bin/run_rppg_qnn.sh" marker)
[[ "${launcher_output}" == "${package_dir}/lib:/existing/lib"$'\n''<marker>' ]] ||
  fail 'launcher added an unexpected library directory'

rm "${package_dir}/bin/rppg_qnn_live"
expect_failure "${tmp_dir}/missing-binary.log" "${package_dir}/bin/run_rppg_qnn.sh"
grep -q 'rppg_qnn_live' "${tmp_dir}/missing-binary.log" ||
  fail 'launcher did not identify the missing executable'

fake_tools="${tmp_dir}/tools"
mkdir -p "${fake_tools}" "${tmp_dir}/synthetic-build"
cat >"${fake_tools}/ctest" <<'EOF'
#!/usr/bin/env bash
if [[ -n ${RPPG_TEST_CTEST_LOG:-} ]]; then
  printf '<%s>\n' "$@" >"${RPPG_TEST_CTEST_LOG}"
fi
EOF
chmod +x "${fake_tools}/ctest"
RPPG_TEST_CTEST_LOG="${tmp_dir}/ctest.log" PATH="${fake_tools}:${PATH}" \
  BUILD_DIR="${tmp_dir}/synthetic-build" "${SMOKE_SCRIPT}"
grep -Fq '<^pipeline$>' "${tmp_dir}/ctest.log" ||
  fail 'no-argument smoke did not select the synthetic pipeline test'

printf 'video' >"${tmp_dir}/sample.avi"
printf 'cascade' >"${tmp_dir}/cascade.xml"
cat >"${tmp_dir}/fake-live" <<'EOF'
#!/usr/bin/env bash
printf '<%s>\n' "$@" >"${RPPG_TEST_LIVE_LOG}"
EOF
chmod +x "${tmp_dir}/fake-live"
RPPG_TEST_LIVE_LOG="${tmp_dir}/live.log" \
  RPPG_QNN_BINARY="${tmp_dir}/fake-live" \
  RPPG_HAAR_CASCADE="${tmp_dir}/cascade.xml" \
  "${SMOKE_SCRIPT}" "${tmp_dir}/sample.avi" "${tmp_dir}/video-output"
grep -q '<--deep>' "${tmp_dir}/live.log" || fail 'video smoke did not configure deep mode'
grep -q '<fake>' "${tmp_dir}/live.log" || fail 'video smoke is not using fake deep wiring'
grep -q "<${tmp_dir}/sample.avi>" "${tmp_dir}/live.log" ||
  fail 'video smoke did not preserve the video path'

cat >"${fake_tools}/cmake" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == --build ]]; then
  mkdir -p "$2"
  printf '#!/usr/bin/env bash\nexit 0\n' >"$2/rppg_qnn_live"
  chmod +x "$2/rppg_qnn_live"
elif [[ ${1:-} == --install ]]; then
  prefix=
  while [[ $# -gt 0 ]]; do
    if [[ $1 == --prefix ]]; then
      prefix=$2
      break
    fi
    shift
  done
  [[ -n ${prefix} ]]
  mkdir -p "${prefix}/bin" "${prefix}/share/rppg-qnn/config"
  cp "${RPPG_TEST_SOURCE_DIR}/packaging/run_rppg_qnn.sh" \
    "${prefix}/bin/run_rppg_qnn.sh"
  cp "${RPPG_TEST_SOURCE_DIR}/README.md" "${prefix}/share/rppg-qnn/README.md"
  cp "${RPPG_TEST_SOURCE_DIR}/config/runtime-defaults.env" \
    "${prefix}/share/rppg-qnn/config/runtime-defaults.env"
  printf '#!/usr/bin/env bash\nexit 0\n' >"${prefix}/bin/rppg_qnn_live"
  chmod +x "${prefix}/bin/rppg_qnn_live" "${prefix}/bin/run_rppg_qnn.sh"
  if [[ -n ${RPPG_TEST_EXTRA_PACKAGE_FILE:-} ]]; then
    printf 'unexpected' >"${prefix}/${RPPG_TEST_EXTRA_PACKAGE_FILE}"
  fi
fi
EOF
cat >"${fake_tools}/file" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "${RPPG_TEST_FILE_OUTPUT}"
EOF
chmod +x "${fake_tools}/cmake" "${fake_tools}/file"

case "$(uname -s):$(uname -m)" in
  Linux:x86_64) native_file_output='ELF 64-bit LSB pie executable, x86-64' ;;
  Linux:aarch64|Linux:arm64) native_file_output='ELF 64-bit LSB pie executable, ARM aarch64' ;;
  Darwin:arm64) native_file_output='Mach-O 64-bit executable arm64' ;;
  Darwin:x86_64) native_file_output='Mach-O 64-bit executable x86_64' ;;
  *) fail 'test does not define the current native executable format' ;;
esac

fake_build="${tmp_dir}/fake native build"
staged_package="${tmp_dir}/stage with spaces/rppg package"
mkdir -p "${staged_package}/models" "${staged_package}/python"
printf 'stale' >"${staged_package}/models/stale.pth"
printf 'stale' >"${staged_package}/python/stale.py"
RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT="${native_file_output}" \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${fake_build}" STAGE_DIR="${staged_package}" \
  "${BUILD_SCRIPT}" native
[[ $(<"${fake_build}/.rppg-build-mode") == native ]] ||
  fail 'native build did not persist its mode marker'
[[ ! -e "${staged_package}/models/stale.pth" &&
   ! -e "${staged_package}/python/stale.py" ]] ||
  fail 'staged install retained stale model or Python files'
actual_files=$(CDPATH= cd -- "${staged_package}" && find . -type f | LC_ALL=C sort)
expected_files=$'./bin/rppg_qnn_live\n./bin/run_rppg_qnn.sh\n./share/rppg-qnn/README.md\n./share/rppg-qnn/config/runtime-defaults.env'
[[ ${actual_files} == "${expected_files}" ]] ||
  fail 'staged install did not match the release file whitelist'

outside_stage="${tmp_dir}/outside stage"
mkdir -p "${outside_stage}"
printf 'keep' >"${outside_stage}/keep.txt"
ln -s "${outside_stage}" "${tmp_dir}/stage symlink"
expect_failure "${tmp_dir}/stage-symlink.log" env \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" RPPG_TEST_FILE_OUTPUT="${native_file_output}" \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${tmp_dir}/symlink build" \
  STAGE_DIR="${tmp_dir}/stage symlink" "${BUILD_SCRIPT}" native
[[ $(<"${outside_stage}/keep.txt") == keep ]] ||
  fail 'unsafe STAGE_DIR handling modified a symlink target'

rejected_stage="${tmp_dir}/rejected package"
mkdir -p "${rejected_stage}"
printf 'preserve' >"${rejected_stage}/sentinel.txt"
expect_failure "${tmp_dir}/whitelist.log" env \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" RPPG_TEST_FILE_OUTPUT="${native_file_output}" \
  RPPG_TEST_EXTRA_PACKAGE_FILE='unexpected.txt' PATH="${fake_tools}:${PATH}" \
  BUILD_DIR="${tmp_dir}/whitelist build" STAGE_DIR="${rejected_stage}" \
  "${BUILD_SCRIPT}" native
[[ $(<"${rejected_stage}/sentinel.txt") == preserve &&
   ! -e "${rejected_stage}/unexpected.txt" ]] ||
  fail 'a rejected temporary package changed the previous staged package'

expect_failure "${tmp_dir}/mode-mismatch.log" env \
  AARCH64_TOOLCHAIN_PREFIX=aarch64-linux-gnu- AARCH64_SYSROOT=/tmp/sysroot \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB pie executable, ARM aarch64' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${fake_build}" \
  STAGE_DIR="${tmp_dir}/not-created" "${BUILD_SCRIPT}" aarch64
grep -q 'build mode' "${tmp_dir}/mode-mismatch.log" ||
  fail 'reusing a native build directory as aarch64 was not rejected clearly'

unmarked_cache="${tmp_dir}/unmarked cache"
mkdir -p "${unmarked_cache}"
printf 'CMAKE_BUILD_TYPE:STRING=Release\n' >"${unmarked_cache}/CMakeCache.txt"
expect_failure "${tmp_dir}/unmarked-cache.log" env \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" RPPG_TEST_FILE_OUTPUT="${native_file_output}" \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${unmarked_cache}" \
  STAGE_DIR="${tmp_dir}/not-created-2" "${BUILD_SCRIPT}" native
grep -q 'CMakeCache' "${tmp_dir}/unmarked-cache.log" ||
  fail 'untrusted unmarked CMake cache was not rejected clearly'

expect_failure "${tmp_dir}/wrong-native-arch.log" env \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB executable, IBM S/390' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${tmp_dir}/wrong native build" \
  STAGE_DIR="${tmp_dir}/not-created-3" "${BUILD_SCRIPT}" native
grep -q 'architecture' "${tmp_dir}/wrong-native-arch.log" ||
  fail 'native build accepted an executable for the wrong architecture'

expect_failure "${tmp_dir}/wrong-aarch64-arch.log" env \
  AARCH64_TOOLCHAIN_PREFIX=aarch64-linux-gnu- AARCH64_SYSROOT=/tmp/sysroot \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB executable, x86-64' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${tmp_dir}/wrong aarch64 build" \
  STAGE_DIR="${tmp_dir}/not-created-4" "${BUILD_SCRIPT}" aarch64
grep -q 'AArch64' "${tmp_dir}/wrong-aarch64-arch.log" ||
  fail 'aarch64 build accepted a non-AArch64 executable'

child_environment=$(env -i PATH="${PATH}" bash -c 'source "$1"; env' _ "${DEFAULT_CONFIG}")
grep -qx 'RPPG_QNN_GPU_LIBRARY=libQnnGpu.so' <<<"${child_environment}" ||
  fail 'sourced defaults did not export RPPG_QNN_GPU_LIBRARY to a child process'
grep -qx 'RPPG_OPENCL_LIBRARY=libOpenCL.so' <<<"${child_environment}" ||
  fail 'sourced defaults did not export RPPG_OPENCL_LIBRARY to a child process'
grep -qx 'RPPG_HAAR_CASCADE=/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml' \
  <<<"${child_environment}" ||
  fail 'sourced defaults did not export RPPG_HAAR_CASCADE to a child process'

printf 'packaging behavior passed\n'
