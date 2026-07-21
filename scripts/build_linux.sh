#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {native|aarch64}\n' "${0##*/}" >&2
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

mode=$1
case "${mode}" in
  native|aarch64) ;;
  *)
    printf 'Unsupported build mode %q: expected native or aarch64\n' "${mode}" >&2
    exit 2
    ;;
esac

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd)
build_dir=${BUILD_DIR:-"${source_dir}/build-linux-${mode}"}
stage_dir=${STAGE_DIR:-"${source_dir}/stage/rppg-qnn"}

cmake_args=(
  -S "${source_dir}"
  -B "${build_dir}"
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_TESTING=ON
)

if [[ -n ${CMAKE_PREFIX_PATH:-} ]]; then
  cmake_args+=("-DCMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}")
fi
if [[ -n ${OpenCV_DIR:-} ]]; then
  cmake_args+=("-DOpenCV_DIR=${OpenCV_DIR}")
fi

if [[ "${mode}" == aarch64 ]]; then
  : "${AARCH64_TOOLCHAIN_PREFIX:?AARCH64_TOOLCHAIN_PREFIX is required for aarch64 mode}"
  : "${AARCH64_SYSROOT:?AARCH64_SYSROOT is required for aarch64 mode}"
  cmake_args+=(
    "-DCMAKE_TOOLCHAIN_FILE=${source_dir}/cmake/Toolchains/aarch64-linux.cmake"
  )
fi

printf 'Configuring %s build in %s\n' "${mode}" "${build_dir}"
cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --config Release --parallel

if [[ "${mode}" == native ]]; then
  ctest --test-dir "${build_dir}" -C Release --output-on-failure
fi

cmake --install "${build_dir}" --config Release --prefix "${stage_dir}"
test -x "${stage_dir}/bin/rppg_qnn_live"
test -x "${stage_dir}/bin/run_rppg_qnn.sh"
printf 'Staged %s package at %s\n' "${mode}" "${stage_dir}"
