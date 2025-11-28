import pickle
from pathlib import Path
from .qubo import QUBO

def save_qubo(path: str | Path, qubo: QUBO) -> None:
    path = Path(path)
    with path.open("wb") as f:
        pickle.dump(qubo, f)

def load_qubo(path: str | Path) -> QUBO:
    path = Path(path)
    with path.open("rb") as f:
        return pickle.load(f)