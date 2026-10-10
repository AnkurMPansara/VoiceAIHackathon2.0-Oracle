"""CLI interface for single-seller best time to call inference.

Usage:
    python btc_cli.py

Interactive CLI that:
1. Takes seller ID
2. Loads seller data and past attempts
3. Takes time window (24h format)
4. Runs inference and shows top 10 recommended slots
5. Shows historical best time in that window
"""

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
from btc.model.stats import zero_state, apply_contribution
from btc.features.fourier import fourier, time_to_hours
from btc.model.reward import compute_reward

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
        hours = time_to_hours(call_start)
        try:
            k = (d - 1) // 2
            phi = fourier(hours, k=k)
        except ValueError:
            continue
        state = apply_contribution(state, phi, reward, sigma2)
    return state


def parse_time_24h(time_str):
    """Parse time in 24h format (HH:MM)."""
    try:
        parts = time_str.strip().split(":")
        if len(parts) != 2:
            raise ValueError()
        hour = int(parts[0])
        minute = int(parts[1])
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            raise ValueError()
        return hour, minute
    except (ValueError, IndexError):
        return None


def main():
    print("=" * 80)
    print("BEST TIME TO CALL - CLI Interface")
    print("=" * 80)
    print()

    # Get seller ID
    seller_id = input("Please enter seller glid: ").strip()
    if not seller_id:
        print("Error: Seller ID is required")
        return 1

    # Load data
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

    # Filter attempts for this seller
    seller_attempts = [r for r in raw_attempts if str(r.get("seller_id", "")) == seller_id]
    seller_attempts = seller_attempts[:10]  # Top 10

    print(f"Found {len(seller_attempts)} recent attempts for seller {seller_id}")

    if seller_attempts:
        print("\nRECENT ATTEMPTS:")
        print("-" * 80)
        print(f"{'Call Start (IST)':<25} {'Answered':<10} {'Meeting':<10} {'Disposition':<20}")
        print("-" * 80)
        _IST = timezone(timedelta(hours=5, minutes=30))
        for attempt in seller_attempts:
            call_start = attempt.get("call_start_time")
            if call_start:
                ist = call_start.astimezone(_IST)
                ist_str = ist.strftime("%Y-%m-%d %H:%M")
                answered = "Yes" if attempt.get("answered") else "No"
                meeting = "Yes" if attempt.get("meeting_fixed") else "No"
                disposition = attempt.get("disposition", "UNKNOWN")
                print(f"{ist_str:<25} {answered:<10} {meeting:<10} {disposition:<20}")
        print("-" * 80)

    # Get time window
    print()
    start_time_str = input("Enter start time (HH:MM, 24h format, e.g., 22:20): ").strip()
    end_time_str = input("Enter end time (HH:MM, 24h format, e.g., 6:00): ").strip()

    start_hm = parse_time_24h(start_time_str)
    end_hm = parse_time_24h(end_time_str)

    if start_hm is None or end_hm is None:
        print("Error: Invalid time format. Use HH:MM (24h format)")
        return 1

    start_hour, start_minute = start_hm
    end_hour, end_minute = end_hm

    # Calculate window
    start_minutes = start_hour * 60 + start_minute
    end_minutes = end_hour * 60 + end_minute

    if end_minutes <= start_minutes:
        # Window crosses midnight
        window_minutes = (24 * 60 - start_minutes) + end_minutes
    else:
        window_minutes = end_minutes - start_minutes

    window_hours = window_minutes / 60.0

    print(f"\nWindow: {start_time_str} to {end_time_str} ({window_hours:.1f} hours)")

    # Load model bundle and build state
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

    # Normalize data and build seller state
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

    seller_training = [r for r in training_records if r.get("seller_id") == seller_id]
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

    # Generate candidate hours with 15-minute intervals
    _IST = timezone(timedelta(hours=5, minutes=30))
    interval_minutes = 15
    n_intervals = int(window_minutes / interval_minutes)

    candidate_hours = []
    for i in range(n_intervals + 1):
        current_minutes = start_minutes + i * interval_minutes
        if current_minutes >= 24 * 60:
            current_minutes -= 24 * 60
        hour = current_minutes / 60.0
        candidate_hours.append(hour)

    # Run inference
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

    # Add IST timestamps and sort by reward
    for i, slot in enumerate(candidate_slots):
        current_minutes = start_minutes + i * interval_minutes
        if current_minutes >= 24 * 60:
            current_minutes -= 24 * 60
        slot_hour = current_minutes / 60.0
        slot_minute = current_minutes % 60

        # Use today's date with the slot time
        today = datetime.now().date()
        slot_datetime = datetime.combine(today, datetime.min.time()) + timedelta(
            hours=int(slot_hour),
            minutes=slot_minute
        )
        slot_datetime_ist = slot_datetime.replace(tzinfo=_IST)

        slot["timestamp_ist"] = slot_datetime_ist.strftime("%Y-%m-%d %H:%M:%S IST")
        slot["hour_display"] = f"{int(slot_hour):02d}:{int(slot_minute):02d}"

    # Sort by expected reward (best to worst)
    candidate_slots.sort(key=lambda x: x["expected_reward"], reverse=True)

    # Get top 10
    top_10 = candidate_slots[:10]

    # Find historical best in window
    historical_best = None
    historical_best_reward = float("-inf")

    for record in seller_training:
        call_start = record.get("call_start_time")
        if call_start is None:
            continue

        call_start_ist = call_start.astimezone(_IST)
        call_minutes = call_start_ist.hour * 60 + call_start_ist.minute

        # Check if in window
        in_window = False
        if end_minutes <= start_minutes:  # Crosses midnight
            in_window = call_minutes >= start_minutes or call_minutes < end_minutes
        else:
            in_window = start_minutes <= call_minutes < end_minutes

        if in_window and record.get("meeting_fixed"):
            # Compute simple reward for historical comparison
            try:
                hist_reward = compute_reward(
                    answered=record.get("answered", False),
                    meeting_fixed=record.get("meeting_fixed", False),
                    disposition=record.get("disposition", "UNKNOWN"),
                    reward_config=reward_config,
                )
                if hist_reward > historical_best_reward:
                    historical_best_reward = hist_reward
                    historical_best = {
                        "timestamp_ist": call_start_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                        "hour_display": f"{call_start_ist.hour:02d}:{call_start_ist.minute:02d}",
                        "reward": hist_reward,
                    }
            except ValueError:
                continue

    # Display results
    print("\n" + "=" * 80)
    print("TOP 10 RECOMMENDED SLOTS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Time (IST)':<25} {'Hour':<8} {'Reward':<12} {'Latent Std':<12} {'Pred Std':<12}")
    print("-" * 80)

    for i, slot in enumerate(top_10):
        print(f"{i+1:<6} {slot['timestamp_ist']:<25} {slot['hour_display']:<8} {slot['expected_reward']:<12.6f} {slot['latent_std']:<12.6f} {slot['predictive_std']:<12.6f}")

    print("-" * 80)
    print()

    if best_slot:
        print(f"RECOMMENDED BEST TIME: {best_slot['timestamp_ist']}")
        print(f"  Hour: {best_slot['hour_display']}")
        print(f"  Expected Reward: {best_slot['expected_reward']:.6f}")
    print()

    if historical_best:
        print(f"HISTORICAL BEST IN WINDOW: {historical_best['timestamp_ist']}")
        print(f"  Hour: {historical_best['hour_display']}")
        print(f"  Actual Reward: {historical_best['reward']:.6f}")
    else:
        print("HISTORICAL BEST: No meetings found in this time window")

    print()
    print("=" * 80)
    print(f"Seller: {seller_id} | Cold Start: {is_cold_start} | Prior Weight: {prior_weight:.4f}")
    print(f"Window: {start_time_str} to {end_time_str} | Slots: {len(candidate_slots)}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
