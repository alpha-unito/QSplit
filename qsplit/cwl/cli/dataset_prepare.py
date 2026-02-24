import argparse
import json
import re
from pathlib import Path


def _record_id(record: dict, line_no: int) -> str:
    raw = record.get("id")
    if raw is None:
        raw = record.get("originale_index", line_no)
    value = str(raw).strip()
    return value if value else str(line_no)


def _safe_id(value: str, fallback: int) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return safe if safe else f"row_{fallback}"


def _matrix_from_record(record: dict) -> list[list[float]] | None:
    try:
        dim = int(record.get("dim", 0))
    except (TypeError, ValueError):
        return None
    if dim <= 0:
        return None

    matrix = [[0.0 for _ in range(dim)] for _ in range(dim)]
    for term in record.get("qubo_mat", []):
        if not isinstance(term, (list, tuple)) or len(term) != 3:
            continue
        try:
            i = int(term[0])
            j = int(term[1])
            value = float(term[2])
        except (TypeError, ValueError):
            continue
        if i < 0 or j < 0 or i >= dim or j >= dim:
            continue
        if i <= j:
            matrix[i][j] = value
        else:
            matrix[j][i] = value
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare per-instance matrix CSV files from a JSONL dataset.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--output-dir", default="dataset_matrices")
    parser.add_argument("--manifest", default="dataset_manifest.json")
    args = parser.parse_args()

    if args.max_instances is not None and args.max_instances <= 0:
        raise ValueError("--max-instances must be > 0 when provided.")

    dataset_path = Path(args.dataset_jsonl).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.csv"):
        stale.unlink()

    manifest_items: list[dict] = []
    with dataset_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, start=1):
            if args.max_instances is not None and len(manifest_items) >= args.max_instances:
                break
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue

            matrix = _matrix_from_record(record)
            if matrix is None:
                continue

            rec_id = _record_id(record, line_no)
            safe_id = _safe_id(rec_id, line_no)
            file_index = len(manifest_items)
            matrix_name = f"{file_index:06d}_{safe_id}.csv"
            matrix_path = output_dir / matrix_name
            matrix_path.write_text(
                "\n".join(",".join(f"{value:g}" for value in row) for row in matrix) + "\n",
                encoding="utf-8",
            )

            manifest_items.append(
                {
                    "index": file_index,
                    "line": line_no,
                    "id": rec_id,
                    "safe_id": safe_id,
                    "matrix_csv": matrix_name,
                }
            )

    if not manifest_items:
        raise SystemExit("No valid records processed from dataset.")

    manifest_path = Path(args.manifest).resolve()
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "count": len(manifest_items),
                "items": manifest_items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
