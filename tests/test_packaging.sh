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

expect_exit_two() {
  local output_file=$1
  shift
  local status
  set +e
  "$@" >"${output_file}" 2>&1
  status=$?
  set -e
  [[ ${status} -eq 2 ]] || fail "expected exit 2, got ${status}: $*"
}

tmp_dir=$(mktemp -d "${TMPDIR:-/tmp}/rppg-qnn-packaging.XXXXXX")
trap 'rm -rf "${tmp_dir}"' EXIT

expect_failure "${tmp_dir}/unknown.log" "${BUILD_SCRIPT}" unsupported
grep -q 'expected native or aarch64' "${tmp_dir}/unknown.log" ||
  fail 'unknown build mode did not explain the accepted modes'

expect_failure "${tmp_dir}/prefix.log" env -u AARCH64_CMAKE_TOOLCHAIN_FILE \
  -u AARCH64_TOOLCHAIN_PREFIX \
  AARCH64_SYSROOT=/tmp/sysroot "${BUILD_SCRIPT}" aarch64
grep -q 'AARCH64_TOOLCHAIN_PREFIX' "${tmp_dir}/prefix.log" ||
  fail 'aarch64 build did not require AARCH64_TOOLCHAIN_PREFIX'

expect_failure "${tmp_dir}/sysroot.log" env -u AARCH64_CMAKE_TOOLCHAIN_FILE \
  -u AARCH64_SYSROOT \
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
if [[ -n ${RPPG_TEST_CMAKE_CALLED:-} ]]; then
  printf 'called\n' >>"${RPPG_TEST_CMAKE_CALLED}"
fi
if [[ -n ${RPPG_TEST_CMAKE_ARGS_LOG:-} ]]; then
  {
    printf 'BEGIN\n'
    printf '<%s>\n' "$@"
    printf 'END\n'
  } >>"${RPPG_TEST_CMAKE_ARGS_LOG}"
fi
if [[ -n ${RPPG_TEST_CMAKE_ENV_LOG:-} ]]; then
  printf 'CC=<%s>\nCXX=<%s>\n' "${CC-}" "${CXX-}" >>"${RPPG_TEST_CMAKE_ENV_LOG}"
fi
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
native_cmake_log="${tmp_dir}/native-cmake.log"
mkdir -p "${staged_package}/models" "${staged_package}/python"
printf 'stale' >"${staged_package}/models/stale.pth"
printf 'stale' >"${staged_package}/python/stale.py"
RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT="${native_file_output}" \
  RPPG_TEST_CMAKE_ARGS_LOG="${native_cmake_log}" \
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
if grep -Fqx '<--verbose>' "${native_cmake_log}"; then
  fail 'default build unexpectedly enabled verbose output'
fi

for invalid_verbose in 2 yes; do
  expect_exit_two "${tmp_dir}/verbose-${invalid_verbose}.log" env \
    RPPG_BUILD_VERBOSE="${invalid_verbose}" RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
    RPPG_TEST_FILE_OUTPUT="${native_file_output}" PATH="${fake_tools}:${PATH}" \
    BUILD_DIR="${tmp_dir}/verbose-${invalid_verbose}-build" \
    STAGE_DIR="${tmp_dir}/verbose-${invalid_verbose}-stage" \
    "${BUILD_SCRIPT}" native
  grep -q 'RPPG_BUILD_VERBOSE.*0 or 1' "${tmp_dir}/verbose-${invalid_verbose}.log" ||
    fail 'invalid RPPG_BUILD_VERBOSE rejection was not explicit'
done

expect_exit_two "${tmp_dir}/relative-toolchain.log" env \
  -u AARCH64_TOOLCHAIN_PREFIX -u AARCH64_SYSROOT \
  AARCH64_CMAKE_TOOLCHAIN_FILE=relative/OEToolchainConfig.cmake \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB pie executable, ARM aarch64' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${tmp_dir}/relative toolchain build" \
  STAGE_DIR="${tmp_dir}/relative toolchain stage" "${BUILD_SCRIPT}" aarch64
grep -q 'absolute' "${tmp_dir}/relative-toolchain.log" ||
  fail 'relative external CMake toolchain rejection was not explicit'

expect_exit_two "${tmp_dir}/missing-toolchain.log" env \
  -u AARCH64_TOOLCHAIN_PREFIX -u AARCH64_SYSROOT \
  AARCH64_CMAKE_TOOLCHAIN_FILE="${tmp_dir}/missing/OEToolchainConfig.cmake" \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB pie executable, ARM aarch64' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${tmp_dir}/missing toolchain build" \
  STAGE_DIR="${tmp_dir}/missing toolchain stage" "${BUILD_SCRIPT}" aarch64
grep -q 'readable regular file' "${tmp_dir}/missing-toolchain.log" ||
  fail 'missing external CMake toolchain rejection was not explicit'

toolchain_directory="${tmp_dir}/toolchain-is-directory"
mkdir -p "${toolchain_directory}"
expect_exit_two "${tmp_dir}/directory-toolchain.log" env \
  -u AARCH64_TOOLCHAIN_PREFIX -u AARCH64_SYSROOT \
  AARCH64_CMAKE_TOOLCHAIN_FILE="${toolchain_directory}" \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB pie executable, ARM aarch64' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${tmp_dir}/directory toolchain build" \
  STAGE_DIR="${tmp_dir}/directory toolchain stage" "${BUILD_SCRIPT}" aarch64
grep -q 'readable regular file' "${tmp_dir}/directory-toolchain.log" ||
  fail 'directory external CMake toolchain rejection was not explicit'

external_toolchain_dir="${tmp_dir}/Yocto SDK with spaces"
external_toolchain="${external_toolchain_dir}/OEToolchainConfig.cmake"
external_toolchain_symlink="${tmp_dir}/OEToolchainConfig link.cmake"
external_build="${tmp_dir}/external toolchain build"
external_stage="${tmp_dir}/external toolchain stage"
external_cmake_log="${tmp_dir}/external-toolchain-cmake.log"
external_cmake_env_log="${tmp_dir}/external-toolchain-cmake-env.log"
mkdir -p "${external_toolchain_dir}"
printf 'set(CMAKE_SYSTEM_NAME Linux)\n' >"${external_toolchain}"
ln -s "${external_toolchain}" "${external_toolchain_symlink}"
env -u AARCH64_TOOLCHAIN_PREFIX -u AARCH64_SYSROOT \
  AARCH64_CMAKE_TOOLCHAIN_FILE="${external_toolchain_symlink}" \
  RPPG_BUILD_VERBOSE=1 \
  RPPG_TEST_CMAKE_ARGS_LOG="${external_cmake_log}" \
  RPPG_TEST_CMAKE_ENV_LOG="${external_cmake_env_log}" \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB pie executable, ARM aarch64' \
  CC='aarch64-oe-linux-gcc -mcpu=cortex-a55 --sysroot=/SDK target sysroot' \
  CXX='aarch64-oe-linux-g++ -mcpu=cortex-a55 --sysroot=/SDK target sysroot' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${external_build}" \
  STAGE_DIR="${external_stage}" "${BUILD_SCRIPT}" aarch64
grep -Fqx "<-DCMAKE_TOOLCHAIN_FILE=${external_toolchain_symlink}>" \
  "${external_cmake_log}" ||
  fail 'external CMake toolchain path was not passed as one exact argument'
awk '
  $0 == "BEGIN" { is_build = 0; is_verbose = 0 }
  $0 == "<--build>" { is_build = 1 }
  $0 == "<--verbose>" { is_verbose = 1 }
  $0 == "END" {
    if (is_verbose && !is_build) bad = 1
    if (is_verbose && is_build) found = 1
  }
  END { exit !(found && !bad) }
' "${external_cmake_log}" ||
  fail 'only the CMake build command should receive --verbose'
grep -Fqx 'CC=<aarch64-oe-linux-gcc -mcpu=cortex-a55 --sysroot=/SDK target sysroot>' \
  "${external_cmake_env_log}" ||
  fail 'compound SDK CC environment value was changed or split'
grep -Fqx 'CXX=<aarch64-oe-linux-g++ -mcpu=cortex-a55 --sysroot=/SDK target sysroot>' \
  "${external_cmake_env_log}" ||
  fail 'compound SDK CXX environment value was changed or split'
[[ -x "${external_stage}/bin/rppg_qnn_live" ]] ||
  fail 'external CMake toolchain mode did not publish the AArch64 package'

legacy_build="${tmp_dir}/legacy prefix build"
legacy_stage="${tmp_dir}/legacy prefix stage"
legacy_cmake_log="${tmp_dir}/legacy-prefix-cmake.log"
env -u AARCH64_CMAKE_TOOLCHAIN_FILE \
  AARCH64_TOOLCHAIN_PREFIX=aarch64-linux-gnu- AARCH64_SYSROOT=/tmp/sysroot \
  RPPG_BUILD_VERBOSE=0 RPPG_TEST_CMAKE_ARGS_LOG="${legacy_cmake_log}" \
  RPPG_TEST_SOURCE_DIR="${SOURCE_DIR}" \
  RPPG_TEST_FILE_OUTPUT='ELF 64-bit LSB pie executable, ARM aarch64' \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${legacy_build}" \
  STAGE_DIR="${legacy_stage}" "${BUILD_SCRIPT}" aarch64
grep -Fqx "<-DCMAKE_TOOLCHAIN_FILE=${SOURCE_DIR}/cmake/Toolchains/aarch64-linux.cmake>" \
  "${legacy_cmake_log}" ||
  fail 'legacy prefix/sysroot mode did not use the repository toolchain file'
if grep -Fqx '<--verbose>' "${legacy_cmake_log}"; then
  fail 'RPPG_BUILD_VERBOSE=0 unexpectedly enabled verbose build output'
fi
[[ -x "${legacy_stage}/bin/rppg_qnn_live" ]] ||
  fail 'legacy prefix/sysroot mode did not publish the AArch64 package'

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

expect_failure "${tmp_dir}/mode-mismatch.log" env -u AARCH64_CMAKE_TOOLCHAIN_FILE \
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

expect_failure "${tmp_dir}/wrong-aarch64-arch.log" env -u AARCH64_CMAKE_TOOLCHAIN_FILE \
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

make_sandbox_source() {
  local destination=$1
  mkdir -p "${destination}/scripts" "${destination}/packaging" "${destination}/config" \
    "${destination}/cmake/Toolchains"
  cp "${BUILD_SCRIPT}" "${destination}/scripts/build_linux.sh"
  cp "${LAUNCHER}" "${destination}/packaging/run_rppg_qnn.sh"
  cp "${DEFAULT_CONFIG}" "${destination}/config/runtime-defaults.env"
  cp "${SOURCE_DIR}/README.md" "${destination}/README.md"
  cp "${SOURCE_DIR}/cmake/Toolchains/aarch64-linux.cmake" \
    "${destination}/cmake/Toolchains/aarch64-linux.cmake"
  chmod +x "${destination}/scripts/build_linux.sh" \
    "${destination}/packaging/run_rppg_qnn.sh"
}

source_parent_root="${tmp_dir}/source parent boundary"
source_parent_repo="${source_parent_root}/repo"
source_parent_build="${tmp_dir}/source parent external build"
make_sandbox_source "${source_parent_repo}"
mkdir -p "${source_parent_build}"
printf 'source-safe' >"${source_parent_repo}/source.sentinel"
printf 'build-safe' >"${source_parent_build}/build.sentinel"
cmake_called="${tmp_dir}/source-parent-cmake-called"
expect_exit_two "${tmp_dir}/source-parent-boundary.log" env \
  RPPG_TEST_CMAKE_CALLED="${cmake_called}" RPPG_TEST_SOURCE_DIR="${source_parent_repo}" \
  RPPG_TEST_FILE_OUTPUT="${native_file_output}" PATH="${fake_tools}:${PATH}" \
  BUILD_DIR="${source_parent_build}" STAGE_DIR="${source_parent_root}" \
  "${source_parent_repo}/scripts/build_linux.sh" native
[[ ! -e "${cmake_called}" && $(<"${source_parent_repo}/source.sentinel") == source-safe &&
   $(<"${source_parent_build}/build.sentinel") == build-safe ]] ||
  fail 'source-parent STAGE_DIR reached CMake or changed a protected sentinel'

build_parent_root="${tmp_dir}/build parent boundary"
build_parent_repo="${tmp_dir}/build parent repo"
build_parent_build="${build_parent_root}/build"
make_sandbox_source "${build_parent_repo}"
mkdir -p "${build_parent_build}"
printf 'source-safe' >"${build_parent_repo}/source.sentinel"
printf 'build-safe' >"${build_parent_build}/build.sentinel"
cmake_called="${tmp_dir}/build-parent-cmake-called"
expect_exit_two "${tmp_dir}/build-parent-boundary.log" env \
  RPPG_TEST_CMAKE_CALLED="${cmake_called}" RPPG_TEST_SOURCE_DIR="${build_parent_repo}" \
  RPPG_TEST_FILE_OUTPUT="${native_file_output}" PATH="${fake_tools}:${PATH}" \
  BUILD_DIR="${build_parent_build}" STAGE_DIR="${build_parent_root}" \
  "${build_parent_repo}/scripts/build_linux.sh" native
[[ ! -e "${cmake_called}" && $(<"${build_parent_repo}/source.sentinel") == source-safe &&
   $(<"${build_parent_build}/build.sentinel") == build-safe ]] ||
  fail 'build-parent STAGE_DIR reached CMake or changed a protected sentinel'

common_root="${tmp_dir}/common ancestor boundary"
common_repo="${common_root}/repo"
common_build="${common_root}/build"
make_sandbox_source "${common_repo}"
mkdir -p "${common_build}"
printf 'source-safe' >"${common_repo}/source.sentinel"
printf 'build-safe' >"${common_build}/build.sentinel"
parent_alias="${tmp_dir}/physical parent alias"
ln -s "${tmp_dir}" "${parent_alias}"
common_root_via_alias="${parent_alias}/$(basename -- "${common_root}")"
cmake_called="${tmp_dir}/common-parent-cmake-called"
expect_exit_two "${tmp_dir}/common-parent-boundary.log" env \
  RPPG_TEST_CMAKE_CALLED="${cmake_called}" RPPG_TEST_SOURCE_DIR="${common_repo}" \
  RPPG_TEST_FILE_OUTPUT="${native_file_output}" PATH="${fake_tools}:${PATH}" \
  BUILD_DIR="${common_build}" STAGE_DIR="${common_root_via_alias}" \
  "${common_repo}/scripts/build_linux.sh" native
[[ ! -e "${cmake_called}" && $(<"${common_repo}/source.sentinel") == source-safe &&
   $(<"${common_build}/build.sentinel") == build-safe ]] ||
  fail 'symlink-resolved common-ancestor STAGE_DIR changed protected data'

descendant_root="${tmp_dir}/allowed descendant boundary"
descendant_repo="${descendant_root}/repo"
descendant_build="${tmp_dir}/allowed descendant build"
descendant_stage="${descendant_repo}/stage/package"
make_sandbox_source "${descendant_repo}"
printf 'source-safe' >"${descendant_repo}/source.sentinel"
RPPG_TEST_SOURCE_DIR="${descendant_repo}" RPPG_TEST_FILE_OUTPUT="${native_file_output}" \
  PATH="${fake_tools}:${PATH}" BUILD_DIR="${descendant_build}" \
  STAGE_DIR="${descendant_stage}" "${descendant_repo}/scripts/build_linux.sh" native
[[ $(<"${descendant_repo}/source.sentinel") == source-safe &&
   -x "${descendant_stage}/bin/rppg_qnn_live" ]] ||
  fail 'a safe STAGE_DIR below the source directory was rejected or damaged the source'

printf 'packaging behavior passed\n'
