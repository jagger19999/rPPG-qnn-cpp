#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s {native|aarch64}\n' "${0##*/}" >&2
}

die() {
  printf 'build_linux.sh: %s\n' "$1" >&2
  exit 2
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

if [[ "${mode}" == aarch64 ]]; then
  : "${AARCH64_TOOLCHAIN_PREFIX:?AARCH64_TOOLCHAIN_PREFIX is required for aarch64 mode}"
  : "${AARCH64_SYSROOT:?AARCH64_SYSROOT is required for aarch64 mode}"
fi

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd)
build_dir=${BUILD_DIR:-"${source_dir}/build-linux-${mode}"}
requested_stage_dir=${STAGE_DIR:-"${source_dir}/stage/rppg-qnn"}

mkdir -p "${build_dir}"
build_dir=$(CDPATH= cd -- "${build_dir}" && pwd)
[[ "${build_dir}" != "${source_dir}" ]] || die 'BUILD_DIR cannot be the source directory'

mode_marker="${build_dir}/.rppg-build-mode"
if [[ -e "${build_dir}/CMakeCache.txt" && ! -f "${mode_marker}" ]]; then
  die "CMakeCache.txt exists without a trusted build mode marker; use a fresh BUILD_DIR"
fi
if [[ -e "${mode_marker}" ]]; then
  [[ -f "${mode_marker}" && ! -L "${mode_marker}" ]] ||
    die 'build mode marker must be a regular file'
  recorded_mode=$(<"${mode_marker}")
  [[ "${recorded_mode}" == "${mode}" ]] ||
    die "BUILD_DIR build mode is ${recorded_mode}, not ${mode}; use a separate BUILD_DIR"
else
  marker_temp="${mode_marker}.tmp.$$"
  printf '%s\n' "${mode}" >"${marker_temp}"
  mv -- "${marker_temp}" "${mode_marker}"
fi

stage_parent_input=$(dirname -- "${requested_stage_dir}")
stage_name=$(basename -- "${requested_stage_dir}")
case "${stage_name}" in
  ''|'.'|'..'|'/') die 'STAGE_DIR must name a package directory' ;;
esac
mkdir -p "${stage_parent_input}"
stage_parent=$(CDPATH= cd -- "${stage_parent_input}" && pwd)
stage_dir="${stage_parent}/${stage_name}"
[[ "${stage_dir}" != "${source_dir}" && "${stage_dir}" != "${build_dir}" ]] ||
  die 'STAGE_DIR cannot be the source or build directory'
if [[ -L "${stage_dir}" ]]; then
  die 'STAGE_DIR cannot be a symbolic link'
fi
if [[ -e "${stage_dir}" && ! -d "${stage_dir}" ]]; then
  die 'STAGE_DIR exists and is not a directory'
fi

temporary_stage=
backup_stage=
safe_remove_transient() {
  local path=$1
  case "${path}" in
    "${stage_parent}/.${stage_name}.staging."*|"${stage_parent}/.${stage_name}.previous."*)
      rm -rf -- "${path}"
      ;;
    *)
      printf 'build_linux.sh: refused to remove unsafe transient path: %s\n' "${path}" >&2
      return 1
      ;;
  esac
}

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${backup_stage}" && -e "${backup_stage}" && ! -e "${stage_dir}" ]]; then
    mv -- "${backup_stage}" "${stage_dir}" || status=1
    backup_stage=
  fi
  if [[ -n "${temporary_stage}" && -e "${temporary_stage}" ]]; then
    safe_remove_transient "${temporary_stage}" || status=1
  fi
  if [[ -n "${backup_stage}" && -e "${backup_stage}" ]]; then
    safe_remove_transient "${backup_stage}" || status=1
  fi
  exit "${status}"
}
trap cleanup EXIT

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

built_binary="${build_dir}/rppg_qnn_live"
if [[ ! -x "${built_binary}" && -x "${build_dir}/Release/rppg_qnn_live" ]]; then
  built_binary="${build_dir}/Release/rppg_qnn_live"
fi
[[ -x "${built_binary}" ]] || die "built executable not found under ${build_dir}"
command -v file >/dev/null 2>&1 || die "the 'file' utility is required for architecture checks"

verify_architecture() {
  local binary=$1
  local description
  description=$(file -b -- "${binary}")
  if [[ "${mode}" == aarch64 ]]; then
    [[ "${description}" == *ELF\ 64-bit* && "${description}" == *ARM\ aarch64* ]] ||
      die "AArch64 package requires a Linux ELF AArch64 executable; got: ${description}"
    return
  fi

  case "$(uname -s):$(uname -m)" in
    Linux:x86_64)
      [[ "${description}" == *ELF* && "${description}" == *x86-64* ]] ;;
    Linux:aarch64|Linux:arm64)
      [[ "${description}" == *ELF* && "${description}" == *ARM\ aarch64* ]] ;;
    Darwin:arm64)
      [[ "${description}" == *Mach-O* && "${description}" == *arm64* ]] ;;
    Darwin:x86_64)
      [[ "${description}" == *Mach-O* && "${description}" == *x86_64* ]] ;;
    *)
      die "unsupported native host architecture: $(uname -s) $(uname -m)" ;;
  esac || die "native executable architecture does not match the host; got: ${description}"
}

verify_architecture "${built_binary}"

temporary_stage=$(mktemp -d "${stage_parent}/.${stage_name}.staging.XXXXXX")
cmake --install "${build_dir}" --config Release --prefix "${temporary_stage}"
test -x "${temporary_stage}/bin/rppg_qnn_live"
test -x "${temporary_stage}/bin/run_rppg_qnn.sh"
verify_architecture "${temporary_stage}/bin/rppg_qnn_live"

actual_files=$(CDPATH= cd -- "${temporary_stage}" && find . -type f | LC_ALL=C sort)
expected_files=$'./bin/rppg_qnn_live\n./bin/run_rppg_qnn.sh\n./share/rppg-qnn/README.md\n./share/rppg-qnn/config/runtime-defaults.env'
[[ "${actual_files}" == "${expected_files}" ]] ||
  die "staged package does not match the release file whitelist: ${actual_files}"
[[ -z "$(find "${temporary_stage}" -type l -print -quit)" ]] ||
  die 'staged package must not contain symbolic links'

if [[ -d "${stage_dir}" ]]; then
  backup_stage=$(mktemp -d "${stage_parent}/.${stage_name}.previous.XXXXXX")
  rmdir -- "${backup_stage}"
  mv -- "${stage_dir}" "${backup_stage}"
fi
if ! mv -- "${temporary_stage}" "${stage_dir}"; then
  die 'could not publish the validated staged package'
fi
temporary_stage=
if [[ -n "${backup_stage}" ]]; then
  safe_remove_transient "${backup_stage}"
  backup_stage=
fi

printf 'Staged %s package at %s\n' "${mode}" "${stage_dir}"
