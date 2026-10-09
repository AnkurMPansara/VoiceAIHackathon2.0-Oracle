"""Phase 0 data quality report for the Best Time to Call prediction system.

Implements SRS EVAL-01 (descriptive statistics) and DATA-08 (import reports):
- Distributions of observations per seller, hour-bin support, weekday, segments,
  call status, lead age
- Coverage diagnostics: unknown mappings, missing profiles, label contradictions
- Temporal diagnostics: event delays, duplicate frequency, lead age distribution

This module provides pure functions for generating a comprehensive data quality
report from normalized outcome data and seller profiles.

Modules
-------
generate_phase0_report : Main entry point – full Phase 0 report.
compute_observation_distribution : Observations per seller distribution.
compute_hour_bin_support : Hour-bin support from call_start_time.
compute_segment_distribution : Segment distribution and eligibility.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import numpy as np

from btc.config import ModelConfig, RewardConfig
from btc.data.adapters import _build_import_report
from btc.data.normalization import create_chronological_splits, validate_split_integrity

logger = logging.getLogger(__name__)


# ── Hour-bin definitions ──────────────────────────────────────────────────────

_HOURS_PER_BIN = 15  # minutes

_BIN_LABELS = [
    "00:00-00:15", "00:15-00:30", "00:30-00:45", "00:45-01:00",
    "01:00-01:15", "01:15-01:30", "01:30-01:45", "01:45-02:00",
    "02:00-02:15", "02:15-02:30", "02:30-02:45", "02:45-03:00",
    "03:00-03:15", "03:15-03:30", "03:30-03:45", "03:45-04:00",
    "04:00-04:15", "04:15-04:30", "04:30-04:45", "04:45-05:00",
    "05:00-05:15", "05:15-05:30", "05:30-05:45", "05:45-06:00",
    "06:00-06:15", "06:15-06:30", "06:30-06:45", "06:45-07:00",
    "07:00-07:15", "07:15-07:30", "07:30-07:45", "07:45-08:00",
    "08:00-08:15", "08:15-08:30", "08:30-08:45", "08:45-09:00",
    "09:00-09:15", "09:15-09:30", "09:30-09:45", "09:45-10:00",
    "10:00-10:15", "10:15-10:30", "10:30-10:45", "10:45-11:00",
    "11:00-11:15", "11:15-11:30", "11:30-11:45", "11:45-12:00",
    "12:00-12:15", "12:15-12:30", "12:30-12:45", "12:45-13:00",
    "13:00-13:15", "13:15-13:30", "13:30-13:45", "13:45-14:00",
    "14:00-14:15", "14:15-14:30", "14:30-14:45", "14:45-15:00",
    "15:00-15:15", "15:15-15:30", "15:30-15:45", "15:45-16:00",
    "16:00-16:15", "16:15-16:30", "16:30-16:45", "16:45-17:00",
    "17:00-17:15", "17:15-17:30", "17:30-17:45", "17:45-18:00",
    "18:00-18:15", "18:15-18:30", "18:30-18:45", "18:45-19:00",
    "19:00-19:15", "19:15-19:30", "19:30-19:45", "19:45-20:00",
    "20:00-20:15", "20:15-20:30", "20:30-20:45", "20:45-21:00",
    "21:00-21:15", "21:15-21:30", "21:30-21:45", "21:45-22:00",
    "22:00-22:15", "22:15-22:30", "22:30-22:45", "22:45-23:00",
    "23:00-23:15", "23:15-23:30", "23:30-23:45", "23:45-00:00",
]

_NUM_BINS = len(_BIN_LABELS)


# ── Helper: bin key from datetime ─────────────────────────────────────────────


def _bin_key(dt: datetime) -> str:
    """Compute the 15-minute bin key for a datetime.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.

    Returns
    -------
    str
        Bin key string, e.g. "2026-04-15T10:00".
    """
    minute = dt.minute
    floored_minute = (minute // _HOURS_PER_BIN) * _HOURS_PER_BIN
    return dt.strftime(f"%Y-%m-%dT%H:{floored_minute:02d}")


def _hour_bin_index(dt: datetime) -> int:
    """Compute the 96-bin index for a datetime.

    Parameters
    ----------
    dt : datetime
        Timezone-aware datetime.

    Returns
    -------
    int
        Index in [0, 96).
    """
    hour = dt.hour
    minute = dt.minute
    return hour * 4 + (minute // 15)


# ── compute_observation_distribution ──────────────────────────────────────────


def compute_observation_distribution(data: list[dict]) -> dict:
    """Distribution of attempts per seller.

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'seller_id' field.

    Returns
    -------
    dict
        {
            'counts': dict[str, int],  # seller_id -> count
            'histogram': dict[str, int],  # count_bucket -> num_sellers
            'summary': dict[str, float],  # mean, median, std, min, max
        }
    """
    seller_counts: Counter = Counter()

    for record in data:
        seller_id = record.get("seller_id")
        if seller_id:
            seller_counts[seller_id] += 1

    counts_list = list(seller_counts.values())

    if counts_list:
        counts_array = np.array(counts_list, dtype=np.float64)
        summary = {
            "mean": float(np.mean(counts_array)),
            "median": float(np.median(counts_array)),
            "std": float(np.std(counts_array, ddof=1)) if len(counts_list) > 1 else 0.0,
            "min": int(np.min(counts_array)),
            "max": int(np.max(counts_array)),
            "num_sellers": len(counts_list),
        }

        # Histogram buckets: 0, 1-4, 5-9, 10-19, 20-49, 50-99, 100-499, 500-999, 1000+
        buckets = {
            "1": 0,
            "2-4": 0,
            "5-9": 0,
            "10-19": 0,
            "20-49": 0,
            "50-99": 0,
            "100-499": 0,
            "500-999": 0,
            "1000+": 0,
        }
        for c in counts_list:
            if c == 1:
                buckets["1"] += 1
            elif c <= 4:
                buckets["2-4"] += 1
            elif c <= 9:
                buckets["5-9"] += 1
            elif c <= 19:
                buckets["10-19"] += 1
            elif c <= 49:
                buckets["20-49"] += 1
            elif c <= 99:
                buckets["50-99"] += 1
            elif c <= 499:
                buckets["100-499"] += 1
            elif c <= 999:
                buckets["500-999"] += 1
            else:
                buckets["1000+"] += 1
    else:
        summary = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0,
            "max": 0,
            "num_sellers": 0,
        }
        buckets = {k: 0 for k in buckets}

    return {
        "counts": dict(seller_counts),
        "histogram": buckets,
        "summary": summary,
    }


# ── compute_hour_bin_support ──────────────────────────────────────────────────


def compute_hour_bin_support(data: list[dict]) -> dict:
    """Hour bin support from call_start_time.

    Computes the distribution of call attempts across 96 fifteen-minute bins,
    including per-bin attempt counts, answer rates, and meeting rates.

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'call_start_time', 'answered', 'meeting_fixed' fields.

    Returns
    -------
    dict
        {
            'bin_counts': dict[str, int],  # bin_label -> count
            'bin_answer_rates': dict[str, float],  # bin_label -> rate
            'bin_meeting_rates': dict[str, float],  # bin_label -> rate
            'bin_sellers': dict[str, int],  # bin_label -> unique sellers
        }
    """
    bin_stats: dict[int, dict[str, Any]] = {
        i: {"n": 0, "answered": 0, "meeting_fixed": 0, "sellers": set()}
        for i in range(_NUM_BINS)
    }

    for record in data:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue

        idx = _hour_bin_index(call_start)
        if idx < 0 or idx >= _NUM_BINS:
            continue

        stats = bin_stats[idx]
        stats["n"] += 1
        if record.get("answered"):
            stats["answered"] += 1
        if record.get("meeting_fixed"):
            stats["meeting_fixed"] += 1
        seller_id = record.get("seller_id")
        if seller_id:
            stats["sellers"].add(seller_id)

    result = {
        "bin_counts": {},
        "bin_answer_rates": {},
        "bin_meeting_rates": {},
        "bin_sellers": {},
    }

    for i in range(_NUM_BINS):
        label = _BIN_LABELS[i]
        stats = bin_stats[i]
        n = stats["n"]
        result["bin_counts"][label] = n
        result["bin_answer_rates"][label] = (
            stats["answered"] / n if n > 0 else 0.0
        )
        result["bin_meeting_rates"][label] = (
            stats["meeting_fixed"] / n if n > 0 else 0.0
        )
        result["bin_sellers"][label] = len(stats["sellers"])

    return result


# ── compute_segment_distribution ──────────────────────────────────────────────


def compute_segment_distribution(data: list[dict]) -> dict:
    """Segment distribution and eligibility.

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'segment', 'seller_id' fields.

    Returns
    -------
    dict
        {
            'segment_counts': dict[str, int],  # segment -> count
            'segment_sellers': dict[str, int],  # segment -> unique sellers
            'segment_eligibility': dict[str, bool],  # segment -> eligible
        }
    """
    segment_data: dict[str, dict[str, Any]] = {}

    for record in data:
        segment = record.get("segment", "UNKNOWN")
        seller_id = record.get("seller_id")

        if segment not in segment_data:
            segment_data[segment] = {
                "n": 0,
                "sellers": set(),
            }

        segment_data[segment]["n"] += 1
        if seller_id:
            segment_data[segment]["sellers"].add(seller_id)

    segment_counts: dict[str, int] = {}
    segment_sellers: dict[str, int] = {}
    segment_eligibility: dict[str, bool] = {}

    for segment, stats in segment_data.items():
        n_sellers = len(stats["sellers"])
        n_attempts = stats["n"]
        segment_counts[segment] = n_attempts
        segment_sellers[segment] = n_sellers
        # TRAIN-01 eligibility: >= 2000 attempts and >= 200 sellers
        segment_eligibility[segment] = (
            n_attempts >= 2000 and n_sellers >= 200
        )

    return {
        "segment_counts": segment_counts,
        "segment_sellers": segment_sellers,
        "segment_eligibility": segment_eligibility,
    }


# ── compute_call_status_distribution ──────────────────────────────────────────


def compute_call_status_distribution(data: list[dict]) -> dict:
    """Distribution of call status fields.

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'answered', 'disposition', 'meeting_fixed' fields.

    Returns
    -------
    dict
        {
            'answer_rate': float,
            'meeting_rate': float,
            'disposition_counts': dict[str, int],
            'disposition_rates': dict[str, float],
        }
    """
    total = len(data)
    if total == 0:
        return {
            "answer_rate": 0.0,
            "meeting_rate": 0.0,
            "disposition_counts": {},
            "disposition_rates": {},
        }

    answered_count = sum(1 for r in data if r.get("answered"))
    meeting_count = sum(1 for r in data if r.get("meeting_fixed"))

    disposition_counter: Counter = Counter()
    for record in data:
        disp = record.get("disposition", "UNKNOWN")
        disposition_counter[disp] += 1

    return {
        "answer_rate": answered_count / total,
        "meeting_rate": meeting_count / total,
        "disposition_counts": dict(disposition_counter),
        "disposition_rates": {
            k: v / total for k, v in disposition_counter.items()
        },
    }


# ── compute_weekday_distribution ──────────────────────────────────────────────


def compute_weekday_distribution(data: list[dict]) -> dict:
    """Distribution of calls across weekdays.

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'call_start_time' field.

    Returns
    -------
    dict
        {
            'weekday_counts': dict[str, int],  # weekday_name -> count
            'weekday_rates': dict[str, float],  # weekday_name -> rate
        }
    """
    weekday_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
    weekday_counter: Counter = Counter()

    for record in data:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue
        weekday_idx = call_start.weekday()  # 0=Monday, 6=Sunday
        weekday_counter[weekday_names[weekday_idx]] += 1

    total = len(data)
    return {
        "weekday_counts": {
            weekday_names[i]: weekday_counter.get(weekday_names[i], 0)
            for i in range(7)
        },
        "weekday_rates": {
            weekday_names[i]: (
                weekday_counter.get(weekday_names[i], 0) / total
            ) if total > 0 else 0.0
            for i in range(7)
        },
    }


# ── compute_temporal_diagnostics ──────────────────────────────────────────────


def compute_temporal_diagnostics(data: list[dict]) -> dict:
    """Temporal diagnostics: event delays, duplicate frequency, lead age.

    Parameters
    ----------
    data : list[dict]
        Normalized data with 'call_start_time', 'lead_sent_time',
        'attempt_number', 'attempt_id' fields.

    Returns
    -------
    dict
        {
            'event_delays': dict,  # summary stats of lead_sent -> call_start delay
            'duplicate_frequency': dict,  # duplicate attempt_id stats
            'lead_age_distribution': dict,  # summary stats of lead age
        }
    """
    delays: list[float] = []
    lead_ages: list[float] = []

    attempt_ids: Counter = Counter()
    for record in data:
        attempt_id = record.get("attempt_id")
        if attempt_id:
            attempt_ids[attempt_id] += 1

        lead_sent = record.get("lead_sent_time")
        call_start = record.get("call_start_time")

        if lead_sent and call_start:
            try:
                if isinstance(lead_sent, datetime) and isinstance(call_start, datetime):
                    delta = (call_start - lead_sent).total_seconds()
                    if delta >= 0:
                        delays.append(delta)
            except (TypeError, ValueError):
                pass

        # Lead age: time from lead creation to call (approximate via lead_sent)
        if lead_sent and call_start:
            try:
                if isinstance(lead_sent, datetime) and isinstance(call_start, datetime):
                    age = (call_start - lead_sent).total_seconds() / 3600.0  # hours
                    if age >= 0:
                        lead_ages.append(age)
            except (TypeError, ValueError):
                pass

    # Duplicate stats
    duplicates = [count for count in attempt_ids.values() if count > 1]
    duplicate_frequencies = {
        "total_attempts": len(attempt_ids),
        "unique_attempts": len(attempt_ids),
        "duplicates": len(duplicates),
        "max_duplicates": max(duplicates) if duplicates else 0,
        "avg_duplicates": (
            sum(duplicates) / len(duplicates) if duplicates else 0.0
        ),
    }

    # Event delay stats
    if delays:
        delays_array = np.array(delays, dtype=np.float64)
        event_delays = {
            "mean_hours": float(np.mean(delays_array)) / 3600.0,
            "median_hours": float(np.median(delays_array)) / 3600.0,
            "p95_hours": float(np.percentile(delays_array, 95)) / 3600.0,
            "min_hours": float(np.min(delays_array)) / 3600.0,
            "max_hours": float(np.max(delays_array)) / 3600.0,
            "n": len(delays),
        }
    else:
        event_delays = {
            "mean_hours": 0.0,
            "median_hours": 0.0,
            "p95_hours": 0.0,
            "min_hours": 0.0,
            "max_hours": 0.0,
            "n": 0,
        }

    # Lead age stats
    if lead_ages:
        lead_ages_array = np.array(lead_ages, dtype=np.float64)
        lead_age_distribution = {
            "mean_hours": float(np.mean(lead_ages_array)),
            "median_hours": float(np.median(lead_ages_array)),
            "p95_hours": float(np.percentile(lead_ages_array, 95)),
            "min_hours": float(np.min(lead_ages_array)),
            "max_hours": float(np.max(lead_ages_array)),
            "n": len(lead_ages),
        }
    else:
        lead_age_distribution = {
            "mean_hours": 0.0,
            "median_hours": 0.0,
            "p95_hours": 0.0,
            "min_hours": 0.0,
            "max_hours": 0.0,
            "n": 0,
        }

    return {
        "event_delays": event_delays,
        "duplicate_frequency": duplicate_frequencies,
        "lead_age_distribution": lead_age_distribution,
    }


# ── compute_coverage_diagnostics ──────────────────────────────────────────────


def compute_coverage_diagnostics(
    data: list[dict],
    sellers: dict,
    import_report: Optional[dict] = None,
) -> dict:
    """Coverage diagnostics: unknown mappings, missing profiles, contradictions.

    Parameters
    ----------
    data : list[dict]
        Normalized data.
    sellers : dict
        Seller profiles keyed by seller_id.
    import_report : dict | None
        Optional import report from DATA-08.

    Returns
    -------
    dict
        {
            'unknown_mappings': int,
            'missing_profiles': int,
            'label_contradictions': int,
            'missing_timestamps': dict[str, int],
        }
    """
    unknown_mappings = 0
    missing_profiles_set: set[str] = set()
    label_contradictions = 0
    missing_timestamps: dict[str, int] = defaultdict(int)

    for record in data:
        # Check for unknown dispositions
        disposition = record.get("disposition", "UNKNOWN")
        if disposition == "UNKNOWN":
            unknown_mappings += 1

        # Check for missing seller profile
        seller_id = record.get("seller_id")
        if seller_id and seller_id not in sellers:
            missing_profiles_set.add(seller_id)

        # Check for missing timestamps
        for ts_field in ["call_start_time", "lead_sent_time", "finalized_at"]:
            if record.get(ts_field) is None:
                missing_timestamps[ts_field] += 1

    # Count contradictions from import report if available
    if import_report:
        exclusions = import_report.get("exclusions_by_reason", {})
        label_contradictions = exclusions.get(
            "contradictory_meeting_fixed", 0
        ) + exclusions.get("contradictory_not_answered", 0)

    return {
        "unknown_mappings": unknown_mappings,
        "missing_profiles": len(missing_profiles_set),
        "label_contradictions": label_contradictions,
        "missing_timestamps": dict(missing_timestamps),
    }


# ── generate_phase0_report ────────────────────────────────────────────────────


def generate_phase0_report(
    normalized_data: list[dict],
    sellers: dict,
    config: ModelConfig,
) -> dict:
    """EVAL-01: Phase 0 descriptive statistics report.

    Produces a comprehensive data quality and distribution report
    from normalized outcome data and seller profiles.

    Parameters
    ----------
    normalized_data : list[dict]
        Normalized outcome records with canonical fields.
    sellers : dict
        Seller profiles keyed by seller_id.
    config : ModelConfig
        Model configuration (used for segment eligibility thresholds).

    Returns
    -------
    dict
        Report with keys:
        - distributions: observations_per_seller, hour_bins, weekdays, segments
        - rates: meeting_rate, answer_rate, disposition_counts
        - coverage: unknown_mappings, missing_profiles, label_contradictions
        - temporal: event_delays, duplicate_frequency, lead_age_distribution
        - summary: input_rows, retained_rows, exclusion_reasons
        - splits: chronological split counts and integrity
    """
    input_rows = len(normalized_data)

    # Distributions
    observations_per_seller = compute_observation_distribution(normalized_data)
    hour_bins = compute_hour_bin_support(normalized_data)
    weekdays = compute_weekday_distribution(normalized_data)
    segments = compute_segment_distribution(normalized_data)

    # Rates
    call_status = compute_call_status_distribution(normalized_data)

    # Coverage
    coverage = compute_coverage_diagnostics(normalized_data, sellers)

    # Temporal
    temporal = compute_temporal_diagnostics(normalized_data)

    # Summary
    summary = {
        "input_rows": input_rows,
        "retained_rows": len(normalized_data),
        "exclusion_reasons": {
            "missing_profiles": coverage["missing_profiles"],
            "unknown_mappings": coverage["unknown_mappings"],
            "label_contradictions": coverage["label_contradictions"],
        },
    }

    # Chronological splits
    splits = create_chronological_splits(normalized_data)
    split_counts = {
        purpose: len(records) for purpose, records in splits.items()
    }
    split_integrity = validate_split_integrity(splits)

    return {
        "distributions": {
            "observations_per_seller": observations_per_seller,
            "hour_bins": hour_bins,
            "weekdays": weekdays,
            "segments": segments,
        },
        "rates": {
            "meeting_rate": call_status["meeting_rate"],
            "answer_rate": call_status["answer_rate"],
            "disposition_counts": call_status["disposition_counts"],
            "disposition_rates": call_status["disposition_rates"],
        },
        "coverage": coverage,
        "temporal": temporal,
        "summary": summary,
        "splits": {
            "counts": split_counts,
            "integrity": split_integrity,
        },
    }


# ── load_and_report (convenience wrapper) ─────────────────────────────────────


def load_and_report(
    attempts_path: str,
    sellers_path: str,
    source_tz: str = "Asia/Kolkata",
    source: str = "indiamart",
) -> dict:
    """Load CSVs and generate Phase 0 report (DATA-08 + EVAL-01).

    Convenience wrapper that loads attempts and sellers CSVs,
    joins them, and produces a Phase 0 report.

    Parameters
    ----------
    attempts_path : str
        Path to attempts CSV.
    sellers_path : str
        Path to sellers CSV.
    source_tz : str
        Source timezone for naive timestamps.
    source : str
        Source identifier.

    Returns
    -------
    dict
        Phase 0 report with DATA-08 import report included.
    """
    from btc.data.adapters import load_attempts_csv, load_sellers_csv

    attempts, import_report = load_attempts_csv(
        attempts_path, source_tz=source_tz, source=source
    )
    sellers = load_sellers_csv(sellers_path)
    joined = attempts  # Already has segment from normalization

    report = generate_phase0_report(joined, sellers, ModelConfig())
    report["import_report"] = import_report
    return report
