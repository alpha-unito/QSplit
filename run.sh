#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

usage() {
  cat <<'USAGE'
Usage: ./run.sh [-n N] [-f PATH] [--from-id ID] [--to-id ID]

  -n, --limit N   Process only the first N valid rows in the JSONL file (default: all)
  -f, --file PATH JSONL input file (default: qubo_max_cut.jsonl or qubo_mac_cut.jsonl)
  --from-id ID    Process only rows with originale_index >= ID (inclusive)
  --to-id ID      Process only rows with originale_index <= ID (inclusive)
USAGE
}

die() { echo "ERROR: $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

LIMIT=0
JSONL_PATH="qubo_max_cut.jsonl"
FROM_ID=""
TO_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--limit)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      LIMIT="$2"
      shift 2
      ;;
    -f|--file)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      JSONL_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --from-id)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      FROM_ID="$2"
      shift 2
      ;;
    --to-id)
      [[ $# -ge 2 ]] || die "Missing value for $1"
      TO_ID="$2"
      shift 2
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

[[ "$LIMIT" =~ ^[0-9]+$ ]] || die "Invalid limit: $LIMIT"
if [[ -n "$FROM_ID" && ! "$FROM_ID" =~ ^-?[0-9]+$ ]]; then
  die "Invalid --from-id: $FROM_ID"
fi
if [[ -n "$TO_ID" && ! "$TO_ID" =~ ^-?[0-9]+$ ]]; then
  die "Invalid --to-id: $TO_ID"
fi

if [[ ! -f "$JSONL_PATH" && -f "qubo_mac_cut.jsonl" ]]; then
  JSONL_PATH="qubo_mac_cut.jsonl"
fi
[[ -f "$JSONL_PATH" ]] || die "JSONL file not found: $JSONL_PATH"

require_cmd "$PYTHON_BIN"
require_cmd streamflow

OUTPUT_DIR="streamflow/cwl/data"
CONFIG_PATH="streamflow/cwl/config.yml"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" - "$JSONL_PATH" "$LIMIT" "$OUTPUT_DIR" "$CONFIG_PATH" "$FROM_ID" "$TO_ID" <<'PY'
import json
import shutil
import subprocess
import sys
from pathlib import Path

jsonl_path = Path(sys.argv[1])
limit = int(sys.argv[2])
output_dir = Path(sys.argv[3])
config_path = Path(sys.argv[4])
from_id_raw = sys.argv[5]
to_id_raw = sys.argv[6]
from_id = int(from_id_raw) if from_id_raw else None
to_id = int(to_id_raw) if to_id_raw else None

def write_matrix(record, fallback_idx):
    idx = record.get("originale_index", fallback_idx)
    try:
        idx = int(idx)
    except Exception:
        idx = fallback_idx

    dim = int(record.get("dim", 0))
    if dim <= 0:
        return None

    mat = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for i, j, v in record.get("qubo_mat", []):
        i = int(i)
        j = int(j)
        val = float(v)
        if i <= j:
            mat[i][j] = val
        else:
            mat[j][i] = val

    output_csv = output_dir / f"maxcut_{idx}.csv"
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
    return output_csv, idx

count = 0
with jsonl_path.open("r", encoding="utf-8") as f:
    for line in f:
        if limit and count >= limit:
            break
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if from_id is not None or to_id is not None:
            try:
                rec_id = int(record.get("originale_index"))
            except Exception:
                continue
            if from_id is not None and rec_id < from_id:
                continue
            if to_id is not None and rec_id > to_id:
                continue
        result = write_matrix(record, count)
        if result is None:
            continue
        output_csv, original_idx = result
        count += 1
        print(f"==> [{count}] Wrote matrix: {output_csv}")
        subprocess.check_call(["streamflow", "run", "streamflow/streamflow.yml"])
        solutions_path = Path("solutions.csv")
        if solutions_path.exists():
            tagged = solutions_path.with_name(f"solutions_{original_idx}.csv")
            shutil.copy2(solutions_path, tagged)
            print(f"==> [{count}] Saved: {tagged}")
        else:
            print(f"WARNING: solutions.csv not found after run #{count}", file=sys.stderr)

if count == 0:
    raise SystemExit("No valid records processed.")
PY
