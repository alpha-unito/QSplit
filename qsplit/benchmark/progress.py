import sys


def progress(stage, message, *, entry=None, index=None, total=None):
    parts = [f"QSPLIT BENCHMARK {stage}"]
    if index is not None:
        parts.append(f"{index}/{total} remaining={total - index}")
    if entry is not None:
        parts.append(f"{entry['problem']}/{entry['instance']}")
        if "run" in entry:
            parts.append(f"run={entry['run'] + 1}")
    parts.append(message)
    print(" | ".join(parts), file=sys.stderr, flush=True)
