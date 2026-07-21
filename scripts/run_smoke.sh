#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir=$(CDPATH= cd -- "${script_dir}/.." && pwd)
build_dir=${BUILD_DIR:-"${source_dir}/build-linux-native"}

if [[ $# -eq 0 ]]; then
  printf 'Running the synthetic pipeline smoke test (no camera required).\n'
  ctest --test-dir "${build_dir}" -C Release -R '^pipeline$' --output-on-failure
  exit 0
fi

if [[ $# -gt 2 ]]; then
  printf 'Usage: %s [VIDEO [OUTPUT_DIR]]\n' "${0##*/}" >&2
  exit 2
fi

video=$1
output_dir=${2:-"${source_dir}/outputs/video-smoke"}
binary=${RPPG_QNN_BINARY:-"${build_dir}/rppg_qnn_live"}

if [[ ! -f "${video}" ]]; then
  printf 'Video file not found: %s\n' "${video}" >&2
  exit 2
fi
if [[ ! -x "${binary}" ]]; then
  printf 'Executable not found: %s\n' "${binary}" >&2
  exit 2
fi
if [[ -n ${RPPG_HAAR_CASCADE:-} && ! -f ${RPPG_HAAR_CASCADE} ]]; then
  printf 'RPPG_HAAR_CASCADE does not name a file: %s\n' "${RPPG_HAAR_CASCADE}" >&2
  exit 2
fi

printf '%s\n' \
  'Video smoke uses --deep fake only to test asynchronous deep wiring.' \
  'It does not run EfficientPhys or prove QNN inference.'
exec "${binary}" \
  --video "${video}" \
  --traditional green \
  --deep fake \
  --output "${output_dir}"
