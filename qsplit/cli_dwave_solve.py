import argparse

from .io_utils import load_qubo, save_qubo
from .qubo import QUBO
from .adapters.dwave.dwave_sa import solve


def main():
    parser = argparse.ArgumentParser(description="Solve a sub-QUBO on D-Wave")
    parser.add_argument("--input-qubo", required=True, help="Input sub-QUBO (.pkl)")
    parser.add_argument("--output-qubo", required=True, help="Output solved sub-QUBO (.pkl)")
    args = parser.parse_args()

    qubo = load_qubo(args.input_qubo)
    if not isinstance(qubo, QUBO):
        raise TypeError(f"Expected QUBO object in {args.input_qubo}, got {type(qubo)}")

    print("\n\n=== D-Wave sub-QUBO ===")
    print(f"file       : {args.input_qubo}")
    print(f"shape      : {qubo.mat.shape}")
    print("mat        :")
    print(qubo.mat)
    print(f"rows_idx   : {qubo.rows_idx}")
    print(f"cols_idx   : {qubo.cols_idx}")
    print(f"offset     : {qubo.offset}")
    print("=======================\n\n")

    df = solve(qubo)

    qubo.solutions = df

    save_qubo(args.output_qubo, qubo)


if __name__ == "__main__":
    main()