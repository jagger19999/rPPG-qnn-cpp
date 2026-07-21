#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
app_root=$(CDPATH= cd -- "${script_dir}/.." && pwd)
binary="${app_root}/bin/rppg_qnn_live"

if [[ ! -x "${binary}" ]]; then
  printf 'rppg_qnn_live is missing or not executable: %s\n' "${binary}" >&2
  exit 127
fi

library_path="${app_root}/lib"
if [[ -n ${QAIRT_TARGET_LIB_DIR:-} ]]; then
  library_path+="${library_path:+:}${QAIRT_TARGET_LIB_DIR}"
fi
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  library_path+="${library_path:+:}${LD_LIBRARY_PATH}"
fi
export LD_LIBRARY_PATH=${library_path}

exec "${binary}" "$@"
