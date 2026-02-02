#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
VENV_DIR="${VENV_DIR:-$HOME/venv/streamflow}"

SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-$HOME/.cache/singularity}"
SINGULARITY_TMPROOT="${SINGULARITY_TMPROOT:-/tmp}"

die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

require_python_312() {
  local ver
  ver="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null)" \
    || die "Unable to run '$PYTHON_BIN' to check version."
  [[ "$ver" == "3.12" ]] || die "Python must be 3.12.x. Found: $ver (via $PYTHON_BIN)"
}

run_py_quiet() {
  local tmp rc
  tmp="$(mktemp)"
  set +e
  "$@" >"$tmp" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    cat "$tmp" >&2
    rm -f "$tmp"
    exit "$rc"
  fi
  rm -f "$tmp"
}

run_quiet_return() {
  local tmp rc
  tmp="$(mktemp)"
  set +e
  "$@" >"$tmp" 2>&1
  rc=$?
  set -e
  if (( rc != 0 )); then
    cat "$tmp" >&2
  fi
  rm -f "$tmp"
  return "$rc"
}

declare -a PIDS=()
declare -A PID_TO_NAME=()
declare -A PID_TO_LOG=()

cleanup_on_exit() {
  local ec=$?
  if (( ec != 0 )); then
  if (( ${#PIDS[@]} > 0 )); then
      echo "A failure occurred. Stopping remaining parallel builds..." >&2
      for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
      done
    fi
  fi
}
trap cleanup_on_exit EXIT

build_one() {
  local sif="$1"
  local def="$2"
  local tmpdir

  [[ -f "$def" ]] || die "Definition file not found: $def"
  rm -f "$sif"

  tmpdir="$(mktemp -d -p "$SINGULARITY_TMPROOT" "singularity_tmp_$(basename "$sif").XXXXXX")"
  trap 'rm -rf "$tmpdir"' RETURN

  echo "==> Building... $sif"
  
  SINGULARITY_CACHEDIR="$SINGULARITY_CACHEDIR" \
  SINGULARITY_TMPDIR="$tmpdir" \
    singularity build --fakeroot "$sif" "$def"
}

build_one_quiet_on_success() {
  local sif="$1"
  local def="$2"
  local tmpdir

  [[ -f "$def" ]] || die "Definition file not found: $def"
  rm -f "$sif"

  tmpdir="$(mktemp -d -p "$SINGULARITY_TMPROOT" "singularity_tmp_$(basename "$sif").XXXXXX")"
  trap 'rm -rf "$tmpdir"' RETURN

  echo "==> Building... $sif"
  if ! run_quiet_return \
      env SINGULARITY_CACHEDIR="$SINGULARITY_CACHEDIR" SINGULARITY_TMPDIR="$tmpdir" \
      singularity build --fakeroot "$sif" "$def"
  then
    die "Build failed: $(basename "$sif")"
  fi
}

require_cmd "$PYTHON_BIN"
require_python_312
require_cmd singularity
require_cmd streamflow

read -r -p "Enter originale_index from qubo_max_cut.jsonl: " ORIGINAL_INDEX
[[ "$ORIGINAL_INDEX" =~ ^[0-9]+$ ]] || die "Invalid originale_index: $ORIGINAL_INDEX"

JSONL_PATH="qubo_max_cut.jsonl"
if [[ ! -f "$JSONL_PATH" && -f "qubo_mac_cut.jsonl" ]]; then
  JSONL_PATH="qubo_mac_cut.jsonl"
fi
[[ -f "$JSONL_PATH" ]] || die "JSONL file not found: $JSONL_PATH"

OUTPUT_DIR="streamflow/cwl/data"
mkdir -p "$OUTPUT_DIR"
OUTPUT_CSV="$OUTPUT_DIR/maxcut_${ORIGINAL_INDEX}.csv"

echo "==> Building matrix for originale_index=$ORIGINAL_INDEX from $JSONL_PATH"
run_py_quiet "$PYTHON_BIN" - "$ORIGINAL_INDEX" "$JSONL_PATH" "$OUTPUT_CSV" "streamflow/cwl/config.yml" <<'PY'
import json
import sys
from pathlib import Path

idx = int(sys.argv[1])
jsonl_path = Path(sys.argv[2])
output_csv = Path(sys.argv[3])
config_path = Path(sys.argv[4])

record = None
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("originale_index") == idx:
            record = obj
            break

if record is None:
    raise SystemExit(f"originale_index {idx} not found in {jsonl_path}")

dim = int(record.get("dim", 0))
if dim <= 0:
    raise SystemExit(f"Invalid dim for originale_index {idx}: {dim}")

mat = [[0.0 for _ in range(dim)] for _ in range(dim)]
for i, j, v in record.get("qubo_mat", []):
    i = int(i)
    j = int(j)
    val = float(v)
    if i <= j:
        mat[i][j] = val
    else:
        mat[j][i] = val

output_csv.parent.mkdir(parents=True, exist_ok=True)
with output_csv.open("w", encoding="utf-8") as f:
    for row in mat:
        f.write(",".join(f"{x:g}" for x in row) + "\n")

lines = config_path.read_text(encoding="utf-8").splitlines()
updated = []
replaced = False
for line in lines:
    if line.strip().startswith("path:"):
        updated.append("  path: data/" + output_csv.name)
        replaced = True
    else:
        updated.append(line)
if not replaced:
    updated.append("input_matrix:")
    updated.append("  class: File")
    updated.append("  path: data/" + output_csv.name)
config_path.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY

echo "==> Wrote matrix: $OUTPUT_CSV"
echo "==> Updated config: streamflow/cwl/config.yml"

mkdir -p "$SINGULARITY_CACHEDIR"
echo "==> Singularity cache dir: $SINGULARITY_CACHEDIR"

echo "==> Using python: $PYTHON_BIN"
echo "==> Venv dir:     $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "==> Creating venv..."
  run_py_quiet "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  echo "==> Venv already exists (reusing)."
fi

echo "==> Activating venv..."
source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
run_py_quiet "$PYTHON_BIN" -m pip install --upgrade pip

echo "==> Installing project (editable) with extra [streamflow]..."
run_py_quiet "$PYTHON_BIN" -m pip install -e ".[streamflow]"

command -v streamflow >/dev/null 2>&1 || die "'streamflow' command not found after installation/activation. Aborting."

build_one_quiet_on_success \
  "streamflow/singularity/images/qsplit-base.sif" \
  "streamflow/singularity/defs/qsplit-base.def"

echo "==> Base image built. Launching remaining builds in parallel..."

start_build_bg() {
  local sif="$1"
  local def="$2"
  local name logfile

  name="$(basename "$sif")"
  logfile="$(mktemp -t "singularity_${name}.XXXXXX.log")"

  (
    build_one "$sif" "$def"
  ) >"$logfile" 2>&1 &

  local pid=$!
  PIDS+=("$pid")
  PID_TO_NAME["$pid"]="$name"
  PID_TO_LOG["$pid"]="$logfile"
  echo "   - Bulding... $name (pid=$pid)"
}

start_build_bg "streamflow/singularity/images/qsplit-sdwave.sif" "streamflow/singularity/defs/qsplit-sdwave.def"
start_build_bg "streamflow/singularity/images/qsplit-sibm.sif"   "streamflow/singularity/defs/qsplit-sibm.def"
start_build_bg "streamflow/singularity/images/qsplit-siqm.sif"   "streamflow/singularity/defs/qsplit-siqm.def"

failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "ERROR: Build failed: ${PID_TO_NAME[$pid]}" >&2
    echo "----- Begin log: ${PID_TO_NAME[$pid]} -----" >&2
    cat "${PID_TO_LOG[$pid]}" >&2 || true
    echo "----- End log: ${PID_TO_NAME[$pid]} -----" >&2
    failed=1

    for other in "${PIDS[@]}"; do
      if [[ "$other" != "$pid" ]]; then
        kill "$other" 2>/dev/null || true
      fi
    done
    break
  else
    echo "==> Completed: ${PID_TO_NAME[$pid]}"
  fi
done

for pid in "${PIDS[@]}"; do
  rm -f "${PID_TO_LOG[$pid]:-}" 2>/dev/null || true
done

(( failed == 0 )) || exit 1

echo "==> All Singularity builds completed successfully."

streamflow run streamflow/streamflow.yml
