import argparse

from qsplit.benchmark.bridge import finalize, prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    initial = commands.add_parser("prepare")
    initial.add_argument("--request", required=True)
    initial.add_argument("--repository")
    initial.add_argument("--output-dir", default="benchmark_bundle")
    final = commands.add_parser("export")
    final.add_argument("--bundle", required=True)
    final.add_argument("--solutions-dir", required=True)
    final.add_argument("--output-dir", default="benchmark_results")
    final.add_argument("--split-method")
    final.add_argument("--aggregate-method")
    final.add_argument("--cut-dim", type=int)
    final.add_argument("--refinement-method")
    final.add_argument("--refinement-loops", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.request, args.output_dir, repository=args.repository))
    else:
        pipeline = {
            key: getattr(args, key)
            for key in ["split_method", "aggregate_method", "cut_dim", "refinement_method", "refinement_loops"]
        }
        print(finalize(args.bundle, args.solutions_dir, args.output_dir, pipeline=pipeline))


if __name__ == "__main__":
    main()
