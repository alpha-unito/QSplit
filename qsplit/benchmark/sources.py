import bz2
import gzip
import hashlib
import lzma
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

DIRECTORIES = {
    slug: f"{i:02d}-{slug}"
    for i, slug in enumerate(
        [
            "marketsplit",
            "labs",
            "birkhoff",
            "steiner",
            "sports",
            "portfolio",
            "independentset",
            "network",
            "routing",
            "topology",
        ],
        1,
    )
}
KINDS = {"instance": "instances", "model": "models", "solution": "solutions", "submission": "submissions"}


def logical_name(path):
    name = Path(path).name
    while Path(name).suffix:
        name = Path(name).stem
    return name


def decompress(path, destination):
    opener = {".xz": lzma.open, ".lzma": lzma.open, ".gz": gzip.open, ".bz2": bz2.open}.get(path.suffix)
    if opener is None:
        return path
    target = destination / path.with_suffix("").name
    target.parent.mkdir(parents=True, exist_ok=True)
    with opener(path, "rb") as src, target.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    return target


def repository_ref(repository):
    result = subprocess.run(["git", "-C", str(repository), "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "local"


def get_problem(slug, *, ref, source_dir, repository=None):
    import qoblib

    if slug not in DIRECTORIES:
        raise ValueError(f"Unknown QOBLIB problem class: {slug}")
    problem = qoblib.get_problem(slug, ref=ref)
    original_files, original_fetch = problem.files, problem.fetch
    root = Path(repository).resolve() if repository is not None else None
    destination = Path(source_dir).resolve()

    def files(*, kind=None):
        if root is None:
            return original_files(kind=kind)
        entries = []
        for k, folder in KINDS.items():
            if kind is not None and kind != k:
                continue
            for path in sorted((root / DIRECTORIES[slug] / folder).rglob("*")):
                if path.is_file():
                    entries.append(
                        {
                            "path": path.relative_to(root).as_posix(),
                            "name": logical_name(path),
                            "filename": path.name,
                            "kind": k,
                        }
                    )
        return entries

    def fetch(name, *, kind=None, decompress=False):
        candidates = [
            e
            for e in files(kind=kind)
            if name in {e["path"], e["filename"], e["name"]} or e["path"].endswith("/" + name)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {slug}/{kind} file for {name!r}, found {len(candidates)}; use its full path"
            )
        entry = candidates[0]
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid repository path in QOBLIB manifest")
        source = root / relative if root else original_fetch(entry["path"], kind=kind)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        if decompress:
            return globals()["decompress"](target, destination / "decompressed" / DIRECTORIES[slug])
        return target

    problem.files = files
    problem.fetch = fetch
    return problem


def load_instance(problem, name):
    if problem.slug != "portfolio":
        return problem.load_instance(name)
    entries = [e for e in problem.files(kind="instance") if name in Path(e["path"]).parts]
    if not entries:
        raise ValueError(f"No portfolio instance directory {name!r}")
    paths = [problem.fetch(e["path"], kind="instance") for e in entries]
    return SimpleNamespace(instance_dir=paths[0].parent, name=name)


def load_model(problem, name, formulation):
    entries = [
        e
        for e in problem.files(kind="model")
        if f"/{formulation}/" in e["path"]
        and e["filename"].endswith((".lp", ".lp.xz", ".lp.gz", ".lp.bz2", ".lp.lzma"))
    ]
    matches = [e for e in entries if name in {e["name"], e["path"], e["filename"]}]
    if not matches:
        matches = [e for e in entries if e["filename"].startswith(name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one LP model for {name!r}, found {len(matches)}; supply its exact model path")
    return problem.load_model(matches[0]["path"], formulation=formulation)


def source_hashes(source_dir):
    root = Path(source_dir)
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
