"""Quick training script — trains the BTC model from CSV files (no DB needed).

Usage:
    python train_from_csv.py
    python train_from_csv.py --data-dir data --output artifacts/model_bundle --dry-run

This script reads the CSV files directly from the data folder, normalizes
the data, fits hierarchical priors, and saves a model bundle.

No PostgreSQL database is required.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from btc.config import default_config, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Train Best Time to Call model from CSV files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory containing CSV data files (default: data)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/development.yaml",
        help="Path to config YAML (default: configs/development.yaml)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for model bundle (default: <data-dir>/model_bundle)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps except saving the bundle",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("train_from_csv")

    # Load config
    logger.info("Loading config from %s", args.config)
    if os.path.isfile(args.config):
        config = load_config(args.config)
    else:
        logger.warning("Config file not found, using defaults")
        config = default_config()

    # Resolve output path
    output_path = args.output or os.path.join(args.data_dir, "model_bundle")

    # Import data trainer
    from btc.data.trainer import train_from_csv_paths

    # Discover CSV files
    from btc.data.trainer import discover_csv_files

    try:
        attempts_path, sellers_path = discover_csv_files(args.data_dir)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Using CSV files:")
    logger.info("  Attempts: %s", attempts_path)
    logger.info("  Sellers:  %s", sellers_path)

    # Train model
    logger.info("Starting training (dry_run=%s)", args.dry_run)
    logger.info("  Output: %s", output_path)

    try:
        report = train_from_csv_paths(
            config=config,
            attempts_path=attempts_path,
            sellers_path=sellers_path,
            output_path=output_path,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Print report
    print("\n" + "=" * 60)
    print("TRAINING REPORT")
    print("=" * 60)
    print(f"  Bundle ID:        {report['bundle_id']}")
    print(f"  Compatibility ID: {report['compatibility_id']}")
    print(f"  Raw attempts:     {report['n_attempts']}")
    print(f"  Normalized:       {report['n_normalized']}")
    print(f"  Splits:           {json.dumps(report['n_splits'], indent=4)}")
    print(f"  Eligible segments:{report['n_segments']}")
    print(f"  Support bins:     {json.dumps(report['n_support_bins'], indent=4)}")
    print(f"  Best hyperparams: {json.dumps(report['hyperparameter_best'], indent=4)}")
    print(f"  Bundle path:      {report['bundle_path']}")
    print(f"  Dry run:          {report['dry_run']}")
    print("=" * 60)

    # Save report to file
    if report["bundle_path"] and not args.dry_run:
        report_file = os.path.join(report["bundle_path"], "training_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Training report saved to %s", report_file)

    return 0


if __name__ == "__main__":
    sys.exit(main())
