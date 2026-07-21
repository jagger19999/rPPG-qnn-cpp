#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory is required}
BUILD_SCRIPT="${SOURCE_DIR}/scripts/build_linux.sh"
SMOKE_SCRIPT="${SOURCE_DIR}/scripts/run_smoke.sh"
LAUNCHER="${SOURCE_DIR}/packaging/run_rppg_qnn.sh"

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
printf '<%s>\n' "$@" >"${RPPG_TEST_CTEST_LOG}"
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

printf 'packaging behavior passed\n'
