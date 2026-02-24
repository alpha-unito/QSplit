import argparse
import json
import shutil
from pathlib import Path


def _flatten(values: list[list[str]]) -> list[str]:
    out: list[str] = []
    for group in values:
        for item in group:
            token = str(item).strip()
            if token:
                out.append(token)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-instance QSplit solutions for a dataset run.")
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--solution-csv", action="append", nargs="+", default=[])
    parser.add_argument("--output-dir", default="solutions_dataset")
    parser.add_argument("--output-manifest", default="dataset_results_manifest.json")
    args = parser.parse_args()

    manifest_path = Path(args.dataset_manifest).resolve()
    dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = dataset_manifest.get("items", [])

    solution_paths = [Path(p).resolve() for p in _flatten(args.solution_csv)]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("*.csv"):
        stale.unlink()

    results: list[dict] = []
    for idx, item in enumerate(items):
        solution_path = solution_paths[idx] if idx < len(solution_paths) else None
        if solution_path is not None and solution_path.exists():
            output_name = f"solutions_{item['safe_id']}.csv"
            target = output_dir / output_name
            shutil.copy2(solution_path, target)
            status = "ok"
        else:
            output_name = None
            status = "missing"
        results.append(
            {
                "index": item["index"],
                "line": item["line"],
                "id": item["id"],
                "safe_id": item["safe_id"],
                "matrix_csv": item["matrix_csv"],
                "solution_csv": output_name,
                "status": status,
            }
        )

    output_manifest = Path(args.output_manifest).resolve()
    output_manifest.write_text(
        json.dumps(
            {
                "dataset": dataset_manifest.get("dataset"),
                "count": len(items),
                "resolved_solutions": sum(1 for row in results if row["status"] == "ok"),
                "items": results,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
