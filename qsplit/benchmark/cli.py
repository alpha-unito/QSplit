import argparse

from qsplit import configuration
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
    final.add_argument("--pipeline-config", help="YAML/JSON file with pipeline provenance supplied by CWL")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.request, args.output_dir, repository=args.repository))
    else:
        pipeline = configuration.load(args.pipeline_config)
        print(finalize(args.bundle, args.solutions_dir, args.output_dir, pipeline=pipeline))


if __name__ == "__main__":
    main()
