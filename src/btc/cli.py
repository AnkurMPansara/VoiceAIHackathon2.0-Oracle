"""Command-line interface for the Best Time to Call system.

Implements SRS §16.3 (Required CLI commands) and §16.2 (WP0 contract/integration).

Provides eight subcommands that correspond to the agent completion contract:

    btc phase0-report          - Generate Phase 0 data quality report
    btc train-priors           - Train prior model bundle
    btc validate-bundle        - Validate a model bundle
    btc backfill-states        - Backfill seller states from outcomes
    btc backtest               - Run predictive backtest
    btc evaluate-ope           - Evaluate one-step OPE
    btc bench                  - Run performance benchmarks
    btc reconcile-state        - Reconcile seller state from ledger

Each subcommand loads configuration, invokes the appropriate module function,
prints structured JSON output, and returns a nonzero exit code on failure.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from btc.config import Config, load_config

# ---------------------------------------------------------------------------
# Structured logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("btc.cli")


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the CLI.

    Sets up a single StreamHandler writing structured JSON log records to
    stderr so that downstream log aggregators can parse them.

    Parameters
    ----------
    level : str
        Logging level string (e.g. "DEBUG", "INFO", "WARNING").
        Default is "INFO".

    References
    ----------
    SRS §13, NFR-08: Structured logs with reason-coded counts.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# JSON output helpers
# ---------------------------------------------------------------------------

def print_json(result: Dict[str, Any], indent: int = 2) -> None:
    """Print a structured JSON result to stdout.

    All CLI command outputs are emitted as a single JSON object on stdout
    so that consumers (CI pipelines, other scripts) can parse them reliably.

    Parameters
    ----------
    result : dict
        The result dictionary to serialise.
    indent : int
        JSON indentation level. Default is 2.

    References
    ----------
    SRS §16.3: Structured output for all commands.
    """
    print(json.dumps(result, indent=indent, default=str))


def exit_with_error(message: str, error_code: str = "CLI_ERROR", exit_code: int = 1) -> None:
    """Print a structured error JSON to stderr and exit with nonzero code.

    Parameters
    ----------
    message : str
        Human-readable error description.
    error_code : str
        Machine-readable error code for downstream consumers.
    exit_code : int
        OS exit code. Default is 1.

    References
    ----------
    SRS §10.2 (API-02): Errors include error_code and message.
    """
    error_payload = {
        "status": "error",
        "error_code": error_code,
        "message": message,
    }
    print(json.dumps(error_payload, indent=2, default=str), file=sys.stderr)
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Shared argument parsing helpers
# ---------------------------------------------------------------------------

def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    """Add the --config argument to a parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend.
    """
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file (SRS §14).",
    )


def _add_output_argument(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    """Add the --output argument to a parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend.
    required : bool
        Whether --output is mandatory. Default is True.
    """
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output JSON report or artefact.",
    )


def _add_dry_run_argument(parser: argparse.ArgumentParser) -> None:
    """Add the --dry-run flag to a parser.

    Parameters
    ----------
    parser : ArgumentParser
        Parser to extend.
    """
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Simulate the operation without writing any state changes (SRS §16.3).",
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def _handle_phase0_report(args: argparse.Namespace) -> Dict[str, Any]:
    """Generate a Phase 0 data quality report.

    Reads the attempts CSV and sellers CSV, computes distributions, missing
    data rates, and quality metrics per SRS §12.1 (EVAL-01).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, attempts_csv, sellers_csv, output.

    Returns
    -------
    dict
        Structured report data.

    References
    ----------
    SRS §12.1 (EVAL-01), §16.3, DATA-08 (import reports).
    """
    logger.info("Starting Phase 0 data quality report")

    config = load_config(args.config)

    try:
        # Import the evaluation module function (WP6 – Evaluation/QA agent)
        from btc.data.adapters import (
            load_attempts_csv,
            load_sellers_csv,
            join_attempts_sellers,
        )
        from btc.evaluation.phase0 import generate_phase0_report  # type: ignore[import-not-found]

        # Load and normalize data
        raw_attempts, import_report = load_attempts_csv(
            args.attempts_csv, source_tz=config.timezone
        )
        sellers = load_sellers_csv(args.sellers_csv)
        normalized_data = join_attempts_sellers(raw_attempts, sellers)

        result = generate_phase0_report(
            normalized_data=normalized_data,
            sellers=sellers,
            config=config.model,
        )

        result["import_report"] = import_report

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            logger.info("Report written to %s", args.output)

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.evaluation.phase0 module not found. "
            "Ensure WP6 evaluation module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="PHASE0_REPORT_FAILED")


def _handle_train_priors(args: argparse.Namespace) -> Dict[str, Any]:
    """Train a prior model bundle.

    Fits hierarchical segment mean curves and a regularised Gaussian seller
    prior per SRS §6 (TRAIN-01 through TRAIN-09).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, data_dir, output, k, dry_run.

    Returns
    -------
    dict
        Training statistics and bundle path.

    References
    ----------
    SRS §6 (TRAIN-01–TRAIN-09), §16.3, §14.
    """
    logger.info("Starting prior training")

    config = load_config(args.config)

    k_override: Optional[int] = args.k

    if k_override is not None:
        config.model.k = k_override
        logger.info("Overriding K to %d", k_override)

    try:
        from btc.data.trainer import train_prior_bundle  # type: ignore[import-not-found]

        result = train_prior_bundle(
            config=config,
            data_dir=args.data_dir,
            output_path=args.output,
            dry_run=args.dry_run,
        )

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            logger.info("Bundle written to %s", args.output)

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.data.trainer module not found. "
            "Ensure WP2 data/training module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="TRAIN_PRIORS_FAILED")


def _handle_validate_bundle(args: argparse.Namespace) -> Dict[str, Any]:
    """Validate a model bundle.

    Checks bundle metadata, dimensions, finite values, symmetry,
    SPD, and checksums per SRS §6 (TRAIN-07, TRAIN-08, MOD-07).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing bundle_path and verbose.

    Returns
    -------
    dict
        Validation report with pass/fail for each check.

    References
    ----------
    SRS §6 (TRAIN-07, MOD-07), §16.3.
    """
    logger.info("Validating bundle at %s", args.bundle_path)

    try:
        from btc.model.bundle import validate_bundle  # type: ignore[import-not-found]

        result = validate_bundle(
            bundle_path=args.bundle_path,
            verbose=args.verbose,
        )

        result["status"] = "ok" if result.get("valid", False) else "validation_failed"
        if not result.get("valid", False):
            exit_with_error(
                f"Bundle validation failed: {result.get('errors', ['Unknown validation error'])}",
                error_code="VALIDATION_FAILED",
            )
        return result

    except ImportError:
        exit_with_error(
            "btc.model.bundle module not found. "
            "Ensure WP1 model module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="VALIDATE_BUNDLE_FAILED")


def _handle_backfill_states(args: argparse.Namespace) -> Dict[str, Any]:
    """Backfill seller states from outcomes.

    Replays finalised outcomes into seller sufficient statistics (A, b, n)
    for a named inactive namespace per SRS §9 (STATE-08).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, bundle_path, namespace, dry_run.

    Returns
    -------
    dict
        Backfill summary with rows processed and state version.

    References
    ----------
    SRS §9 (STATE-08), §16.3, TRAIN-08.
    """
    logger.info("Starting backfill for namespace %s", args.namespace)

    config = load_config(args.config)

    try:
        from btc.store.backfill import backfill_namespace  # type: ignore[import-not-found]

        result = backfill_namespace(
            config=config,
            bundle_path=args.bundle_path,
            namespace=args.namespace,
            dry_run=args.dry_run,
        )

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.store.backfill module not found. "
            "Ensure WP3 persistence module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="BACKFILL_STATES_FAILED")


def _handle_backtest(args: argparse.Namespace) -> Dict[str, Any]:
    """Run predictive backtest.

    Chronological replay of recommendations using only labels available at
    decision time per SRS §12.2 (EVAL-02).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, bundle_path, data_dir, output.

    Returns
    -------
    dict
        Backtest metrics (MSE, NLL, per-bin results).

    References
    ----------
    SRS §12.2 (EVAL-02), §16.3, TRAIN-06.
    """
    logger.info("Starting predictive backtest")

    config = load_config(args.config)

    try:
        from btc.evaluation.backtest import run_backtest  # type: ignore[import-not-found]

        result = run_backtest(
            config=config,
            bundle_path=args.bundle_path,
            data_dir=args.data_dir,
        )

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            logger.info("Backtest results written to %s", args.output)

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.evaluation.backtest module not found. "
            "Ensure WP6 evaluation module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="BACKTEST_FAILED")


def _handle_evaluate_ope(args: argparse.Namespace) -> Dict[str, Any]:
    """Evaluate one-step OPE.

    Computes IPS, SNIPS, and DR estimators on exploration-arm records
    per SRS §12.3 (EVAL-03, EVAL-04).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, bundle_path, decisions_json, output.

    Returns
    -------
    dict
        OPE estimates with ESS, weight stats, and confidence intervals.

    References
    ----------
    SRS §12.3 (EVAL-03, EVAL-04), §16.3, EXP-02.
    """
    logger.info("Starting one-step OPE evaluation")

    config = load_config(args.config)

    try:
        from btc.evaluation.ope import evaluate_ope  # type: ignore[import-not-found]

        result = evaluate_ope(
            config=config,
            bundle_path=args.bundle_path,
            decisions_path=args.decisions_json,
        )

        if args.output:
            Path(args.output).write_text(
                json.dumps(result, indent=2, default=str), encoding="utf-8"
            )
            logger.info("OPE results written to %s", args.output)

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.evaluation.ope module not found. "
            "Ensure WP6 evaluation module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="OPE_EVALUATION_FAILED")


def _handle_bench(args: argparse.Namespace) -> Dict[str, Any]:
    """Run performance benchmarks.

    Measures recommendation and outcome-processing latency against the
    SRS §13 NFR targets (NFR-01 through NFR-06).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, workers, duration_seconds.

    Returns
    -------
    dict
        Benchmark results with p50/p95/p99 latency, throughput, and errors.

    References
    ----------
    SRS §13 (NFR-01–NFR-06), §16.3, T22.
    """
    logger.info("Starting performance benchmarks")

    config = load_config(args.config)

    try:
        from btc.service.benchmarks import run_benchmarks  # type: ignore[import-not-found]

        result = run_benchmarks(
            config=config,
            workers=args.workers,
            duration_seconds=args.duration_seconds,
        )

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.service.benchmarks module not found. "
            "Ensure WP5 API/experiment or WP6 evaluation module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="BENCHMARK_FAILED")


def _handle_reconcile_state(args: argparse.Namespace) -> Dict[str, Any]:
    """Reconcile seller state from ledger.

    Rebuilds seller sufficient statistics from the authoritative attempt
    ledger to bound numerical drift per SRS §9 (STATE-05, STATE-08).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments containing config, namespace, dry_run.

    Returns
    -------
    dict
        Reconciliation summary with sellers processed and state version.

    References
    ----------
    SRS §9 (STATE-05, STATE-08), §16.3, MOD-05.
    """
    logger.info("Starting state reconciliation for namespace %s", args.namespace)

    config = load_config(args.config)

    try:
        from btc.store.reconciliation import reconcile_namespace  # type: ignore[import-not-found]

        result = reconcile_namespace(
            config=config,
            namespace=args.namespace,
            dry_run=args.dry_run,
        )

        result["status"] = "ok"
        return result

    except ImportError:
        exit_with_error(
            "btc.store.reconciliation module not found. "
            "Ensure WP3 persistence module is implemented.",
            error_code="MODULE_NOT_FOUND",
        )
    except FileNotFoundError as exc:
        exit_with_error(str(exc), error_code="FILE_NOT_FOUND")
    except Exception as exc:
        exit_with_error(str(exc), error_code="RECONCILE_STATE_FAILED")


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the root argument parser with all subcommands.

    Returns
    -------
    ArgumentParser
        Fully configured parser ready for ``parse_args``.
    """
    parser = argparse.ArgumentParser(
        prog="btc",
        description="Best Time to Call — CLI for the Dynamic Retry Engine (SRS §16.3)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG-level) logging.",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Available commands (SRS §16.3).",
    )

    # ── phase0-report ──────────────────────────────────────────────────
    p_phase0 = subparsers.add_parser(
        "phase0-report",
        help="Generate Phase 0 data quality report (SRS §12.1, EVAL-01).",
        description=(
            "Analyses attempts and sellers CSVs, computes distributions, "
            "missing data rates, and quality metrics. Outputs a structured "
            "JSON report."
        ),
    )
    _add_config_argument(p_phase0)
    p_phase0.add_argument(
        "--attempts-csv",
        type=str,
        required=True,
        help="Path to the attempts CSV file.",
    )
    p_phase0.add_argument(
        "--sellers-csv",
        type=str,
        required=True,
        help="Path to the sellers CSV file.",
    )
    _add_output_argument(p_phase0)
    p_phase0.set_defaults(handler=_handle_phase0_report)

    # ── train-priors ───────────────────────────────────────────────────
    p_train = subparsers.add_parser(
        "train-priors",
        help="Train prior model bundle (SRS §6, TRAIN-01–TRAIN-09).",
        description=(
            "Fits hierarchical segment mean curves and a regularised "
            "Gaussian seller prior. Produces an immutable model bundle."
        ),
    )
    _add_config_argument(p_train)
    p_train.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing training CSVs.",
    )
    _add_output_argument(p_train)
    p_train.add_argument(
        "--k",
        type=int,
        default=None,
        help="Override K (basis order). Candidates: 2, 3, 4.",
    )
    _add_dry_run_argument(p_train)
    p_train.set_defaults(handler=_handle_train_priors)

    # ── validate-bundle ────────────────────────────────────────────────
    p_validate = subparsers.add_parser(
        "validate-bundle",
        help="Validate a model bundle (SRS §6, TRAIN-07, MOD-07).",
        description=(
            "Checks bundle metadata, dimensions, finite values, symmetry, "
            "SPD, and checksums on load."
        ),
    )
    p_validate.add_argument(
        "--bundle-path",
        type=str,
        required=True,
        help="Path to the model bundle directory.",
    )
    p_validate.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Include detailed per-check results in the output.",
    )
    p_validate.set_defaults(handler=_handle_validate_bundle)

    # ── backfill-states ────────────────────────────────────────────────
    p_backfill = subparsers.add_parser(
        "backfill-states",
        help="Backfill seller states from outcomes (SRS §9, STATE-08).",
        description=(
            "Replays finalised outcomes into seller sufficient statistics "
            "(A, b, n) for a named inactive namespace before activation."
        ),
    )
    _add_config_argument(p_backfill)
    p_backfill.add_argument(
        "--bundle-path",
        type=str,
        required=True,
        help="Path to the model bundle directory.",
    )
    p_backfill.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="Target namespace name for the backfill.",
    )
    _add_dry_run_argument(p_backfill)
    p_backfill.set_defaults(handler=_handle_backfill_states)

    # ── backtest ───────────────────────────────────────────────────────
    p_backtest = subparsers.add_parser(
        "backtest",
        help="Run predictive backtest (SRS §12.2, EVAL-02).",
        description=(
            "Chronological replay of recommendations using only labels "
            "available at decision time. Reports MSE, NLL, and per-bin metrics."
        ),
    )
    _add_config_argument(p_backtest)
    p_backtest.add_argument(
        "--bundle-path",
        type=str,
        required=True,
        help="Path to the model bundle directory.",
    )
    p_backtest.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing evaluation CSVs.",
    )
    _add_output_argument(p_backtest)
    p_backtest.set_defaults(handler=_handle_backtest)

    # ── evaluate-ope ───────────────────────────────────────────────────
    p_ope = subparsers.add_parser(
        "evaluate-ope",
        help="Evaluate one-step OPE (SRS §12.3, EVAL-03, EVAL-04).",
        description=(
            "Computes IPS, SNIPS, and DR estimators on exploration-arm "
            "records with logged probabilities."
        ),
    )
    _add_config_argument(p_ope)
    p_ope.add_argument(
        "--bundle-path",
        type=str,
        required=True,
        help="Path to the model bundle directory.",
    )
    p_ope.add_argument(
        "--decisions-json",
        type=str,
        required=True,
        help="Path to decisions JSON file with logged actions and probabilities.",
    )
    _add_output_argument(p_ope)
    p_ope.set_defaults(handler=_handle_evaluate_ope)

    # ── bench ──────────────────────────────────────────────────────────
    p_bench = subparsers.add_parser(
        "bench",
        help="Run performance benchmarks (SRS §13, NFR-01–NFR-06).",
        description=(
            "Measures recommendation and outcome-processing latency "
            "against SFR targets. Reports p50/p95/p99, throughput, and errors."
        ),
    )
    _add_config_argument(p_bench)
    p_bench.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads. Default: 4.",
    )
    p_bench.add_argument(
        "--duration-seconds",
        type=int,
        default=900,
        help="Benchmark duration in seconds. Default: 900 (15 min).",
    )
    p_bench.set_defaults(handler=_handle_bench)

    # ── reconcile-state ────────────────────────────────────────────────
    p_reconcile = subparsers.add_parser(
        "reconcile-state",
        help="Reconcile seller state from ledger (SRS §9, STATE-05, STATE-08).",
        description=(
            "Rebuilds seller sufficient statistics from the authoritative "
            "attempt ledger to bound numerical drift."
        ),
    )
    _add_config_argument(p_reconcile)
    p_reconcile.add_argument(
        "--namespace",
        type=str,
        required=True,
        help="Target namespace to reconcile.",
    )
    _add_dry_run_argument(p_reconcile)
    p_reconcile.set_defaults(handler=_handle_reconcile_state)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """Top-level CLI entry point.

    Parses arguments, dispatches to the appropriate subcommand handler,
    prints structured JSON output, and returns an exit code.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments. If None, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        0 on success, nonzero on failure.

    References
    ----------
    SRS §16.3: Each command has --help, explicit paths, nonzero exit on
    failure, structured output, and --dry-run where applicable.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Configure logging based on --verbose flag
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level)

    # Dispatch to the appropriate handler
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1

    start_time = time.time()

    try:
        result: Dict[str, Any] = handler(args)
        elapsed = round(time.time() - start_time, 3)

        output: Dict[str, Any] = {
            "command": args.command,
            "status": result.get("status", "ok"),
            "elapsed_seconds": elapsed,
        }
        output.update(result)

        print_json(output)
        return 0

    except KeyboardInterrupt:
        exit_with_error("Interrupted by user", error_code="INTERRUPTED", exit_code=130)
    except Exception as exc:
        logger.exception("Unhandled exception in command %s", args.command)
        exit_with_error(str(exc), error_code="UNHANDLED_ERROR", exit_code=1)


if __name__ == "__main__":
    main()
