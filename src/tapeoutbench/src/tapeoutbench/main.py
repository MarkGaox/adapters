"""
Main entry point for the TapeoutBench adapter. Do not modify any of the existing flags.

Constructs the Adapter class defined in adapter.py and calls run() to generate tasks
in the Harbor format at the configured output directory.
"""

import argparse
import sys
from pathlib import Path

from .adapter import TapeoutBenchAdapter, TapeoutBenchError

# Default output dir: <repo>/datasets/<adapter_id>
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "datasets" / "tapeoutbench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated tasks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate only the first N tasks",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tasks",
    )
    parser.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Only generate these task IDs",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="TapeoutBench package from the authors: unpacked directory or .tar.gz",
    )
    parser.add_argument(
        "--source-sha256",
        default=None,
        help="Expected sha256 of the .tar.gz package (pins the benchmark version)",
    )
    args = parser.parse_args()

    adapter = TapeoutBenchAdapter(
        args.output_dir,
        overwrite=args.overwrite,
        limit=args.limit,
        task_ids=args.task_ids,
        source=args.source,
        source_sha256=args.source_sha256,
    )

    try:
        written = adapter.run()
    except TapeoutBenchError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"Generated {len(written)} task(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
