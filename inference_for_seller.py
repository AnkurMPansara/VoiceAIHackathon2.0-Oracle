"""Interactive single-seller inference script.

Usage:
    python inference_for_seller.py

Prompts for seller ID and current time, loads data from CSV files,
and displays predictions with 15-minute intervals over a 6-hour window.
"""

from __future__ import annotations

import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from btc.config import RewardConfig
from btc.data.adapters import load_attempts_csv, load_sellers_csv
from btc.data.normalization import normalize_outcome, create_chronological_splits
from btc.evaluation.inference import predict_for_time_slots
from btc.model.bundle import load_bundle
from btc.model.posterior import compute_posterior
from btc.model.reward import compute_reward
from btc.model.stats import zero_state, apply_contribution

GLOBAL_SEGMENT_KEY = '["global"]'


def _parent_key(key):
    import json
    try:
        p = json.loads(key)
    except (TypeError, ValueError):
        return GLOBAL_SEGMENT_KEY if key != GLOBAL_SEGMENT_KEY else None
    if not isinstance(p, list) or not p:
        return GLOBAL_SEGMENT_KEY
    if len(p) > 1:
        return json.dumps(p[:-1])
    return None if p == ["global"] else GLOBAL_SEGMENT_KEY


def _find_segment_prior(segment_key, metadata, segment_keys, segment_mu0, segment_sigma0, segment_lambda0, segment_eta0):
    import json
    import numpy as np

    def _make_raw_prior(mu0, Sigma0, Lambda0, eta0, d):
        eta0 = (Lambda0 @ mu0).astype(np.float64)
        prior = type('Prior', (), {})()
        prior.mu0 = mu0
        prior.Sigma0 = Sigma0
        prior.Lambda0 = Lambda0
        prior.eta0 = eta0
        prior.d = d
        return prior

    d = segment_mu0.shape[1]

    def _try_idx(key):
        for i, k in enumerate(segment_keys):
            if k == key:
                return i
        return None

    key = segment_key
    while key is not None:
        idx = _try_idx(key)
        if idx is not None:
            return _make_raw_prior(
                segment_mu0[idx], segment_sigma0[idx],
                segment_lambda0[idx], segment_eta0[idx], d,
            )
        key = _parent_key(key)
    return None


def _make_global_prior(bundle_arrays, d):
    import numpy as np

    def _make_raw_prior(mu0, Sigma0, Lambda0, eta0, d):
        eta0 = (Lambda0 @ mu0).astype(np.float64)
        prior = type('Prior', (), {})()
        prior.mu0 = mu0
        prior.Sigma0 = Sigma0
        prior.Lambda0 = Lambda0
        prior.eta0 = eta0
        prior.d = d
        return prior

    try:
        return _make_raw_prior(
            bundle_arrays["global_mu0"],
            bundle_arrays["global_sigma0"],
            bundle_arrays["global_lambda0"],
            bundle_arrays["global_eta0"],
            d,
        )
    except KeyError as exc:
        raise ValueError(f"Bundle is missing global prior array: {exc}") from exc


def build_seller_state(seller_records, reward_config, sigma2, d):
    state = zero_state(d)
    for record in seller_records:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue
        try:
            reward = compute_reward(
                answered=record.get("answered", False),
                meeting_fixed=record.get("meeting_fixed", False),
                disposition=record.get("disposition", "UNKNOWN"),
                reward_config=reward_config,
            )
        except ValueError:
            continue
        from btc.features.fourier import fourier, time_to_hours
        hours = time_to_hours(call_start)
        try:
            k = (d - 1) // 2
            phi = fourier(hours, k=k)
        except ValueError:
            continue
        state = apply_contribution(state, phi, reward, sigma2)
    return state


def main():
    print("=" * 70)
    print("BEST TIME TO CALL - Single Seller Inference")
    print("=" * 70)

    seller_id = input("Please enter seller glid: ").strip()
    if not seller_id:
        print("Error: Seller ID is required")
        return 1

    current_time_str = input(
        "Enter current time (YYYY-MM-DD HH:MM) [default: now]: "
    ).strip()

    if current_time_str:
        try:
            current_time = datetime.strptime(current_time_str, "%Y-%m-%d %H:%M")
            current_time = current_time.replace(tzinfo=timezone.utc)
        except ValueError:
            print("Error: Invalid time format. Use YYYY-MM-DD HH:MM")
            return 1
    else:
        current_time = datetime.now(timezone.utc)

    bundle_path = os.path.join(os.path.dirname(__file__), "artifacts", "model_bundle")
    data_dir = os.path.join(os.path.dirname(__file__), "data")

    if not os.path.isdir(bundle_path):
        print(f"Error: Model bundle not found at {bundle_path}")
        print("Please train the model first using: python train_from_csv.py")
        return 1

    print("\nLoading data...")

    # Load CSV files
    attempts_path, sellers_path = None, None
    for filename in sorted(os.listdir(data_dir)):
        if filename.lower().endswith(".csv"):
            if "seller" in filename.lower():
                sellers_path = os.path.join(data_dir, filename)
            elif "attempt" in filename.lower() or "call" in filename.lower():
                attempts_path = os.path.join(data_dir, filename)

    if not attempts_path or not sellers_path:
        print(f"Error: Could not find CSV files in {data_dir}")
        return 1

    raw_attempts, _ = load_attempts_csv(attempts_path)
    sellers = load_sellers_csv(sellers_path)

    normalized_data = []
    for raw in raw_attempts:
        try:
            record = normalize_outcome(raw, sellers)
            normalized_data.append(record)
        except ValueError:
            continue

    splits = create_chronological_splits(normalized_data)
    training_records = (
        splits.get("prior_fit", []) +
        splits.get("warmup", []) +
        splits.get("validation", [])
    )

    seller_training = [r for r in training_records if r.get("seller_id") == str(seller_id)]

    # Load model bundle
    bundle_metadata, bundle_arrays = load_bundle(bundle_path)
    k = int(bundle_metadata.get("k", 4))
    d = int(bundle_metadata.get("d", 2 * k + 1))
    sigma2 = float(bundle_metadata.get("sigma2", 0.06))

    rp = bundle_metadata.get("reward_params", {})
    reward_config = RewardConfig(
        w_meeting=float(rp.get("w_meeting", 1.0)),
        w_answered=float(rp.get("w_answered", 0.1)),
        c_dial=float(rp.get("c_dial", 0.02)),
        w_not_interested=float(rp.get("w_not_interested", 0.0)),
        sigma2=sigma2,
    )

    # Build seller state
    state = build_seller_state(seller_training, reward_config, sigma2, d)

    # Find segment prior
    segment_keys = bundle_metadata.get("segment_keys", [])
    segment_mu0 = bundle_arrays.get("segment_mu0")
    segment_sigma0 = bundle_arrays.get("segment_sigma0")
    segment_lambda0 = bundle_arrays.get("segment_lambda0")
    segment_eta0 = bundle_arrays.get("segment_eta0")

    segment_key = GLOBAL_SEGMENT_KEY
    if seller_training:
        for record in seller_training:
            seg = record.get("segment", GLOBAL_SEGMENT_KEY)
            if seg != GLOBAL_SEGMENT_KEY:
                segment_key = seg
                break

    segment_prior = _find_segment_prior(
        segment_key, bundle_metadata, segment_keys,
        segment_mu0, segment_sigma0, segment_lambda0, segment_eta0,
    )
    if segment_prior is None:
        segment_prior = _make_global_prior(bundle_arrays, d)

    posterior = compute_posterior(state, segment_prior, sigma2)
    prior_weight = float(posterior.prior_weight) if hasattr(posterior, "prior_weight") else 1.0
    is_cold_start = state.n == 0

    # Generate time slots
    _IST = timezone(timedelta(hours=5, minutes=30))
    window_hours = 6.0
    interval_minutes = 15

    start_hour = current_time.astimezone(_IST).hour + current_time.astimezone(_IST).minute / 60.0
    n_intervals = int(window_hours * 60 / interval_minutes)
    candidate_hours = []
    for i in range(n_intervals + 1):
        slot_hour = start_hour + i * interval_minutes / 60.0
        slot_hour = slot_hour % 24
        candidate_hours.append(slot_hour)

    # Run pure inference
    support_bins = bundle_metadata.get("support_bins", {})
    candidate_slots, best_slot = predict_for_time_slots(
        candidate_hours=candidate_hours,
        state=state,
        segment_prior=segment_prior,
        sigma2=sigma2,
        k=k,
        support_bins=support_bins,
        segment_key=segment_key,
    )

    # Format time slots with IST timestamps
    for i, slot in enumerate(candidate_slots):
        slot_time = current_time + timedelta(minutes=i * interval_minutes)
        slot_time_ist = slot_time.astimezone(_IST)
        slot["timestamp_ist"] = slot_time_ist.strftime("%Y-%m-%d %H:%M:%S IST")

    if best_slot:
        best_idx = candidate_hours.index(best_slot["hour"])
        best_time = current_time + timedelta(minutes=best_idx * interval_minutes)
        best_time_ist = best_time.astimezone(_IST)
        best_slot["timestamp_ist"] = best_time_ist.strftime("%Y-%m-%d %H:%M:%S IST")

    # Filter historical attempts within window
    window_start_ist = current_time.astimezone(_IST)
    window_end_ist = window_start_ist + timedelta(hours=window_hours)
    window_start_minutes = window_start_ist.hour * 60 + window_start_ist.minute
    window_end_minutes = window_end_ist.hour * 60 + window_end_ist.minute
    if window_end_minutes >= 24 * 60:
        window_end_minutes -= 24 * 60

    historical_attempts = []
    for record in seller_training:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue
        call_start_ist = call_start.astimezone(_IST)
        call_minutes = call_start_ist.hour * 60 + call_start_ist.minute

        in_window = False
        if window_start_minutes <= window_end_minutes:
            in_window = window_start_minutes <= call_minutes < window_end_minutes
        else:
            in_window = call_minutes >= window_start_minutes or call_minutes < window_end_minutes

        if in_window:
            historical_attempts.append({
                "call_start_time_ist": call_start_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                "hour": float(call_start_ist.hour),
                "minute": call_start_ist.minute,
                "answered": bool(record.get("answered", False)),
                "meeting_fixed": bool(record.get("meeting_fixed", False)),
                "disposition": record.get("disposition", "UNKNOWN"),
            })

    n_historical_attempts = len(seller_training)
    n_historical_meetings = sum(1 for r in seller_training if r.get("meeting_fixed", False))
    n_historical_attempts_in_window = len(historical_attempts)
    historical_meeting_rate = (
        n_historical_meetings / n_historical_attempts if n_historical_attempts > 0 else None
    )

    print("\n" + "=" * 70)
    print("INFERENCE RESULTS")
    print("=" * 70)
    print(f"Seller ID:          {seller_id}")
    print(f"Current Time:       {current_time.astimezone(_IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    print(f"Window:             {window_hours} hours")
    print(f"Cold Start:         {is_cold_start}")
    print(f"Prior Weight:       {prior_weight:.4f}")
    print(f"Historical Attempts:{n_historical_attempts}")
    print(f"In Window:          {n_historical_attempts_in_window}")
    print(f"Historical Meetings:{n_historical_meetings}")
    if historical_meeting_rate is not None:
        print(f"Historical Rate:    {historical_meeting_rate:.4f}")
    print()

    print("CANDIDATE SLOTS (15-min intervals):")
    print("-" * 70)
    print(f"{'Time (IST)':<25} {'Hour':<8} {'Reward':<12} {'Latent Std':<12} {'Pred Std':<12} {'Supported':<10}")
    print("-" * 70)

    for slot in candidate_slots:
        supported = "Yes" if slot['is_supported'] else "No"
        print(f"{slot['timestamp_ist']:<25} {slot['hour']:<8.2f} {slot['expected_reward']:<12.6f} {slot['latent_std']:<12.6f} {slot['predictive_std']:<12.6f} {supported:<10}")

    print("-" * 70)
    if best_slot:
        print(f"\nBEST SLOT: {best_slot['timestamp_ist']} (Hour: {best_slot['hour']:.2f}, Reward: {best_slot['expected_reward']:.6f})")
    print()

    if historical_attempts:
        print("HISTORICAL ATTEMPTS (within 6-hour window):")
        print("-" * 70)
        print(f"{'Call Start (IST)':<28} {'Hour':<8} {'Answered':<10} {'Meeting':<10} {'Disposition':<20}")
        print("-" * 70)
        for m in historical_attempts[:30]:
            answered = "Yes" if m['answered'] else "No"
            meeting = "Yes" if m['meeting_fixed'] else "No"
            disposition = m.get('disposition', 'UNKNOWN')
            print(f"{m['call_start_time_ist']:<28} {m['hour']:<8.0f} {answered:<10} {meeting:<10} {disposition:<20}")
        if len(historical_attempts) > 30:
            print(f"... and {len(historical_attempts) - 30} more")
        print("-" * 70)

    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
