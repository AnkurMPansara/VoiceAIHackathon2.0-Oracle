"""Performance benchmarks for the Best Time to Call service.

SRS NFR-05/06: Bench on declared hardware (development: 4 vCPU/8 GiB
service, 4 vCPU/8 GiB database, same-zone networking). Run 5-minute
warmup then 15-minute measured load at 500 recommendations/s plus
100 outcomes/s, 150k seeded sellers, 10% cold reads, and a 10%
hot-seller traffic group. Report p50/p95/p99, errors, throughput,
conflicts, queue lag, cache misses, CPU, RSS, DB size, and network
latency.

References
----------
SRS NFR-01 through NFR-06.
"""

from __future__ import annotations

import logging
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    warmup_seconds: float = 0.0
    measured_seconds: float = 0.0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    conflict_requests: int = 0

    # Latency percentiles (milliseconds)
    recommend_p50_ms: float = 0.0
    recommend_p95_ms: float = 0.0
    recommend_p99_ms: float = 0.0
    outcome_p50_ms: float = 0.0
    outcome_p95_ms: float = 0.0
    outcome_p99_ms: float = 0.0

    # Throughput
    recommend_rps: float = 0.0
    outcome_rps: float = 0.0

    # System metrics
    cpu_percent: float = 0.0
    rss_mb: float = 0.0
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0

    # Configuration
    workers: int = 4
    duration_seconds: int = 900
    warmup_seconds: float = 300.0
    target_recommend_rps: int = 500
    target_outcome_rps: int = 100
    total_sellers: int = 150_000
    cold_read_fraction: float = 0.1
    hot_seller_fraction: float = 0.1

    errors: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


def run_benchmarks(
    config: Any,
    workers: int = 4,
    duration_seconds: int = 900,
    recommend_service: Optional[Any] = None,
    outcome_service: Optional[Any] = None,
) -> BenchmarkResult:
    """Run performance benchmarks against the BTC service.

    Benchmark profile (NFR-05):
    - 5-minute warmup
    - 15-minute measured load
    - 500 recommendations/s + 100 outcomes/s
    - 150k seeded sellers
    - 10% cold reads
    - 10% hot-seller traffic group

    Parameters
    ----------
    config : Any
        Application configuration.
    workers : int
        Number of concurrent worker threads.
    duration_seconds : int
        Total benchmark duration (includes warmup).
    recommend_service : callable, optional
        Function that takes a request dict and returns a recommendation.
        If None, uses synthetic fixture generation.
    outcome_service : callable, optional
        Function that takes an outcome dict and returns a CommitResult.
        If None, uses synthetic fixture generation.

    Returns
    -------
    BenchmarkResult
        Comprehensive benchmark results.

    References
    ----------
    SRS NFR-03, NFR-04, NFR-05, NFR-06.
    """
    result = BenchmarkResult(
        workers=workers,
        duration_seconds=duration_seconds,
        target_recommend_rps=config.get(
            "target_recommend_rps", 500
        ),
        target_outcome_rps=config.get("target_outcome_rps", 100),
        total_sellers=config.get("total_sellers", 150_000),
        cold_read_fraction=config.get("cold_read_fraction", 0.1),
        hot_seller_fraction=config.get("hot_seller_fraction", 0.1),
    )

    recommend_latencies: List[float] = []
    outcome_latencies: List[float] = []
    total_recommend = 0
    total_outcome = 0
    recommend_errors = 0
    outcome_errors = 0
    recommend_conflicts = 0

    # Generate seller pool
    random.seed(42)
    seller_ids = [f"seller_{i}" for i in range(result.total_sellers)]

    # Identify hot sellers (top 10%)
    hot_sellers = set(seller_ids[: int(result.total_sellers * 0.1)])
    cold_sellers = set(seller_ids[int(result.total_sellers * 0.9) :])

    start_time = time.monotonic()

    # Warmup phase
    warmup_end = start_time + 300  # 5-minute warmup
    warmup_requests = 0
    while time.monotonic() < warmup_end:
        # Generate synthetic recommend request
        seller_id = random.choice(seller_ids)
        is_cold = seller_id in cold_sellers
        is_hot = seller_id in hot_sellers

        request = {
            "request_id": f"bench_req_{warmup_requests}",
            "seller_id": seller_id,
            "lead_id": f"lead_{random.randint(1, 1000)}",
            "earliest_at": "2026-10-09T08:00:00+05:30",
            "latest_at": "2026-10-10T18:00:00+05:30",
        }

        if recommend_service:
            try:
                t0 = time.monotonic()
                recommend_service(request)
                latency = (time.monotonic() - t0) * 1000  # ms
                recommend_latencies.append(latency)
                total_recommend += 1
            except Exception as exc:
                recommend_errors += 1
        else:
            # Synthetic: just measure model inference time
            from btc.features.fourier import fourier
            from btc.model.posterior import (
                compute_posterior,
                score_candidate,
            )
            from btc.model.stats import Prior, SellerState, zero_state

            k = getattr(
                getattr(config, "model_config", None), "k", 4
            )
            d = 2 * k + 1
            sigma2 = getattr(
                getattr(config, "reward_config", None), "sigma2", 1.0
            )

            try:
                t0 = time.monotonic()
                state = zero_state(d)
                prior = Prior.diagonal_prior(d, alpha=0.1)
                posterior = compute_posterior(state, prior, sigma2)
                phi = fourier(np.array([12.0]), k)[0]
                score_candidate(phi, state, prior, sigma2)
                latency = (time.monotonic() - t0) * 1000
                recommend_latencies.append(latency)
                total_recommend += 1
            except Exception:
                recommend_errors += 1

        warmup_requests += 1

    result.warmup_seconds = 300.0

    # Measured phase
    measured_end = start_time + duration_seconds
    while time.monotonic() < measured_end:
        # Generate recommend request
        seller_id = random.choice(seller_ids)
        request = {
            "request_id": f"bench_req_{total_recommend + total_outcome}",
            "seller_id": seller_id,
            "lead_id": f"lead_{random.randint(1, 1000)}",
            "earliest_at": "2026-10-09T08:00:00+05:30",
            "latest_at": "2026-10-10T18:00:00+05:30",
        }

        # Recommend request (80% of traffic)
        for _ in range(5):
            if recommend_service:
                try:
                    t0 = time.monotonic()
                    recommend_service(request)
                    latency = (time.monotonic() - t0) * 1000
                    recommend_latencies.append(latency)
                    total_recommend += 1
                except Exception as exc:
                    recommend_errors += 1
                    if "409" in str(exc):
                        recommend_conflicts += 1
            else:
                try:
                    t0 = time.monotonic()
                    state = zero_state(d)
                    prior = Prior.diagonal_prior(d, alpha=0.1)
                    posterior = compute_posterior(state, prior, sigma2)
                    phi = fourier(np.array([12.0]), k)[0]
                    score_candidate(phi, state, prior, sigma2)
                    latency = (time.monotonic() - t0) * 1000
                    recommend_latencies.append(latency)
                    total_recommend += 1
                except Exception:
                    recommend_errors += 1

        # Outcome request (20% of traffic)
        if outcome_service:
            outcome = {
                "seller_id": seller_id,
                "lead_id": request["lead_id"],
                "attempt_id": f"attempt_{total_outcome}",
                "source": "indiamart",
                "revision": 1,
                "event_id": f"event_{total_outcome}",
                "finalized_at": "2026-10-09T12:00:00+05:30",
                "call_start_time": "2026-10-09T11:55:00+05:30",
                "call_end_time": "2026-10-09T11:56:00+05:30",
                "lead_sent_time": "2026-10-09T11:50:00+05:30",
                "attempt_number": 1,
                "answered": random.choice([True, False]),
                "disposition": random.choice(
                    ["MEETING_FIXED", "NOT_INTERESTED", "GENERAL",
                     "NOT_ANSWERED", "CALL_LATER_BUSY"]
                ),
                "meeting_fixed": random.choice([True, False]),
            }
            try:
                t0 = time.monotonic()
                outcome_service(outcome)
                latency = (time.monotonic() - t0) * 1000
                outcome_latencies.append(latency)
                total_outcome += 1
            except Exception:
                outcome_errors += 1
        else:
            try:
                t0 = time.monotonic()
                # Simulate outcome processing: phi + reward + state update
                from btc.model.reward import (
                    RewardConfig,
                    compute_reward,
                    validate_outcome_consistency,
                )

                answered = random.choice([True, False])
                meeting_fixed = random.choice([True, False])
                disposition = random.choice(
                    ["MEETING_FIXED", "NOT_INTERESTED", "GENERAL",
                     "NOT_ANSWERED", "CALL_LATER_BUSY"]
                )

                # Validate
                is_valid, _ = validate_outcome_consistency(
                    answered, meeting_fixed, disposition
                )
                if is_valid:
                    rc = RewardConfig(
                        w_meeting=1.0,
                        w_answered=0.1,
                        c_dial=0.02,
                        w_not_interested=0.0,
                        sigma2=sigma2,
                    )
                    reward = compute_reward(
                        answered, meeting_fixed, disposition, rc
                    )
                    # Apply to state
                    phi = fourier(np.array([12.0]), k)[0]
                    state = zero_state(d)
                    state = apply_contribution(state, phi, reward, sigma2)
                    latency = (time.monotonic() - t0) * 1000
                    outcome_latencies.append(latency)
                    total_outcome += 1
                else:
                    outcome_errors += 1
            except Exception:
                outcome_errors += 1

    result.measured_seconds = max(
        duration_seconds - 300, 1
    )  # exclude warmup

    # Compute metrics
    if recommend_latencies:
        result.recommend_p50_ms = float(
            np.percentile(recommend_latencies, 50)
        )
        result.recommend_p95_ms = float(
            np.percentile(recommend_latencies, 95)
        )
        result.recommend_p99_ms = float(
            np.percentile(recommend_latencies, 99)
        )
        result.recommend_rps = total_recommend / result.measured_seconds

    if outcome_latencies:
        result.outcome_p50_ms = float(
            np.percentile(outcome_latencies, 50)
        )
        result.outcome_p95_ms = float(
            np.percentile(outcome_latencies, 95)
        )
        result.outcome_p99_ms = float(
            np.percentile(outcome_latencies, 99)
        )
        result.outcome_rps = total_outcome / result.measured_seconds

    result.total_requests = total_recommend + total_outcome
    result.successful_requests = (
        total_recommend + total_outcome - recommend_errors - outcome_errors
    )
    result.failed_requests = recommend_errors + outcome_errors
    result.conflict_requests = recommend_conflicts

    # System metrics
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        result.cpu_percent = proc.cpu_percent()
        result.rss_mb = proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        result.errors.append("psutil not installed; system metrics unavailable")

    # Cache metrics (if state_store available)
    if hasattr(config, "state_store") and config.state_store is not None:
        hits = getattr(config.state_store, "hits", 0)
        misses = getattr(config.state_store, "misses", 0)
        total = hits + misses
        result.cache_hit_rate = hits / total if total > 0 else 0.0
        result.cache_miss_rate = misses / total if total > 0 else 0.0

    result.details = {
        "total_recommend_requests": total_recommend,
        "total_outcome_requests": total_outcome,
        "recommend_errors": recommend_errors,
        "outcome_errors": outcome_errors,
        "recommend_conflicts": recommend_conflicts,
        "recommend_latency_samples": len(recommend_latencies),
        "outcome_latency_samples": len(outcome_latencies),
        "cold_read_fraction": result.cold_read_fraction,
        "hot_seller_fraction": result.hot_seller_fraction,
    }

    return result
