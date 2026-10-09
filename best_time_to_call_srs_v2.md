# Best Time to Call and Dynamic Retry Engine

## Software Requirements Specification 2.0

**Date:** 9 October 2026  
**Target:** IndiaMART outbound calling infrastructure  
**Audience:** Engineering, data science, QA, product owners, and LLM coding agents  
**Status:** Ready for implementation and shadow testing; live activation requires the release inputs in §18.  
**Supersedes:** Implementation Requirements v1.1. Requirements in this document take precedence over its pseudocode and defaults.

The system recommends a concrete future call timestamp and, after an outcome, recommends the next permissible retry. It learns a time-of-day reward curve from population segments and seller history. It does not place calls, reserve dialer capacity, or use an LLM at runtime. LLM agents may implement the system against the contracts and tests below.

**Feasibility verdict:** Technically feasible as a staged system. The statistical benefit, achievable production latency, and infrastructure footprint require measurement. The previous specification is a useful design draft but is not safe to implement literally. Sparse seller observations, ambiguous time handling, non-atomic deduplication, and incorrect propensity assumptions are the principal gaps.

## 1. Review of the supplied specification

The supplied document reports 530,748 attempts and 149,363 sellers for April–September 2026, approximately 3.55 attempts per seller. These are source-document figures, not independently validated data. No raw dataset, production traces, or benchmark results accompany this review.

| Finding | Assessment and required resolution |
|---|---|
| Small linear model | Feasible. Updating a fixed-size information matrix is O(d²); posterior factorization is O(d³). Neither proves network or end-to-end latency. |
| Sparse seller history | Population pooling is essential. The average alone does not establish the fraction of sellers below five attempts. Report the actual distribution. Nine seller coefficients cannot be independently identified from three or four observations without strong assumptions. |
| Unproven performance claims | Replace “all NFRs met” and estimated Redis overhead with measurable targets. Include profile lookup, serialization, logging, concurrency, and persistence in service benchmarks. |
| Discounted updates | `gamma < 1` makes the supplied update dependent on arrival order. It does not commute. Version 2 uses `gamma = 1`; forgetting is deferred. |
| Prior replacement | Existing statistics are reusable only under compatible feature, reward, likelihood, and history definitions. Changing K or reward weights is not a hot swap. |
| Prior estimation | “Method of moments / EM” is not an implementation contract. Version 2 specifies a reproducible pooled-mean and regularized covariance baseline. Learned between-seller covariance is deferred. |
| Noise estimate | Marginal reward variance is not residual variance. A numerical value near 0.06 and a fourfold adaptation claim are not established by the supplied evidence. Calibrate prospectively in chronological validation. |
| Idempotency | A separate SETNX followed by state mutation can lose an update after a crash; a seven-day deduplication expiry can later count it twice. Use a durable transaction covering the attempt revision and statistics. |
| Time representation | A fractional hour does not identify a date. Requests require lead identity, eligibility/deadline timestamps, and calendar constraints. Responses return timezone-aware timestamps. |
| Exploration | A softmax probability is not the probability of a Thompson-sampling argmax. Use explicit uniform sampling for the first release and log its actual probability. |
| Counterfactual evaluation | Estimated historical propensities do not remove unmeasured confounding or create missing action support. Historical OPE is diagnostic; randomized evaluation is required for a credible lift claim. |
| Retry state | Attempt count needs a seller–lead scope; daily caps need a seller scope across all leads. Retries must use call end time, respect deadlines, and never schedule into the past. |
| Retry experiment | Changing timing and retry rules simultaneously prevents clean attribution. Run timing and retry experiments separately in the first release. |
| Peaks | K does not guarantee two useful observed peaks. A unimodal or flat curve may have no secondary peak; return null rather than inventing one. |
| Code contradictions | The scalar Fourier example returns a 2-D array despite its contract. Smoothness penalties disagree on the intercept. Both are fixed below. |
| Memory accounting | The 456-byte claim describes only one proposed packed record. It excludes deduplication, history, retry state, logs, caches, and allocator overhead. These must have separate budgets. |

## 2. Scope and normative language

“SHALL” denotes a mandatory requirement. “MAY” denotes an optional capability. Defaults explicitly marked **development only** allow implementation and simulation; they do not assert business approval.

### 2.1 First-release scope

1. Normalize historical call data and versioned seller profiles; report quality and coverage.
2. Train hierarchical segment mean curves and a regularized Gaussian seller prior.
3. Maintain seller sufficient statistics from finalized, revisioned outcomes.
4. Generate timestamp recommendations subject to eligibility, calendar, support, and deadline constraints.
5. Produce idempotent retry recommendations and integrate through an external scheduler contract.
6. Support shadow mode, deterministic treatment, and explicitly randomized exploration.
7. Provide predictive backtesting, limited one-step OPE, operational telemetry, and experiment reports.

### 2.2 Deferred or excluded

Full covariance EM, Thompson sampling, online forgetting, weekday/context model features, automatic business-policy tuning, duration-dependent reward costs, globally optimal retry sequences, capacity optimization, and automatic call placement are outside release 1. Context fields are logged for diagnostics and future model versions. The double-call experiment contract is specified but disabled until separately activated.

### 2.3 Release stages

| Stage | Required behavior |
|---|---|
| Build | Synthetic fixtures, all interfaces, executable tests, offline tools, no production dependency required. |
| Shadow | Read real events and compute/log proposals; never submit scheduling commands. |
| Timing experiment | Keep the approved retry policy identical across arms; vary eligible scheduling choices only. |
| Retry experiment | Freeze the timing policy; vary approved retry rules in a separate experiment. |
| General availability | Activate only after correctness, operations, and business gates pass. |

## 3. Architecture and responsibility boundaries

**Implementation baseline:** Python, NumPy/SciPy, FastAPI/Pydantic, PostgreSQL, and optional Redis. Dependency versions SHALL be pinned in a lockfile after compatibility testing. Agents SHALL not use unpinned “latest” dependencies.

PostgreSQL is an explicit change from v1.1: it is the authoritative store for attempt revisions, seller state, decisions, and an outbox. This makes correction handling and crash recovery concrete. Redis is an optional read cache, not the only durable copy. The added transactional work makes the original two-millisecond update target a benchmark question.

| Component | Responsibility |
|---|---|
| Data adapter | Map source fields, parse timestamps, preserve source provenance, quarantine invalid rows. |
| Prior trainer | Select hyperparameters, create immutable bundles and validation reports. |
| Model core | Pure Fourier, reward, posterior, uncertainty, and scoring functions. |
| Outcome processor | Atomically apply unique revisions to durable seller state and publish an outbox record. |
| Recommendation service | Resolve context, generate candidates, select action, durably log the decision. |
| Retry engine | Apply the deterministic decision table and scheduling constraints. |
| Scheduler adapter | Reserve capacity, enforce aggregate caps, enqueue once, cancel obsolete work, report actual execution. |
| Experiment evaluator | Join assignment, decisions, execution, and finalized outcomes; report denominators and uncertainty. |

The existing dialer remains responsible for dispatch. A returned recommendation is not a reservation. The scheduler SHALL revalidate restrictions immediately before placing a call. Multiple leads for the same seller SHALL share one cap and reservation authority.

## 4. Canonical data contracts

### 4.1 Common rules

**DATA-01.** IDs are nonempty strings, maximum 128 UTF-8 bytes. Source integer IDs are converted losslessly to decimal strings. JSON numbers SHALL NOT be used for identifiers.

**DATA-02.** API timestamps SHALL be ISO 8601 with an explicit offset. Store instants in UTC; derive business date/hour in `Asia/Kolkata`. Reject naive API timestamps. Historical naive timestamps may be localized only using an explicitly configured source timezone recorded in the import report.

**DATA-03.** Durations are nonnegative integer seconds. All numeric model inputs SHALL be finite. Unknown JSON fields SHALL be rejected on versioned public schemas. Missing is not equivalent to false.

### 4.2 Finalized attempt outcome

| Field | Type and meaning |
|---|---|
| `seller_id`, `lead_id`, `attempt_id` | Required strings; attempt ID globally unique within the source namespace. |
| `source` | Required string identifying the producer; `(source, attempt_id)` is the durable uniqueness key. |
| `revision` | Positive integer, increasing for corrections to the same attempt. |
| `event_id` | Required unique delivery identifier, retained for audit. |
| `finalized_at` | Required timestamp when this outcome became available to consumers. |
| `call_start_time`, `call_end_time` | Required timestamps, start ≤ end. Actual dial start defines model time. |
| `lead_sent_time` | Required timestamp, ≤ call start. |
| `attempt_number` | Positive integer within `(seller_id, lead_id)`; includes all actual dials, including unanswered calls. |
| `answered` | Required boolean, derived from the authoritative call-status mapping. |
| `disposition` | `MEETING_FIXED`, `NOT_INTERESTED`, `GENERAL`, `CALL_LATER_BUSY`, `NOT_ANSWERED`, or `UNKNOWN`. |
| `meeting_fixed` | Required boolean; SHALL agree with the mapped meeting disposition for release 1. |
| `requested_callback_at` | Nullable timestamp explicitly requested by the seller; never inferred from the model. |
| `decision_id` | Nullable ID linking to the scheduling decision. Missing means unavailable for randomized OPE. |
| `duration_s` | Nullable; if present, agrees with call end minus start within one second. |
| `dialer_version`, `source_bucket` | Nullable strings for diagnostics. |

**DATA-04.** `meeting_fixed=true` requires `answered=true` and `disposition=MEETING_FIXED`. `NOT_ANSWERED` requires `answered=false`. `CALL_LATER_BUSY` SHALL NOT itself determine whether the call was answered. Unknown disposition may update the model when answered and meeting labels are valid, but creates no automatic retry.

**DATA-05.** Revisions SHALL NOT change seller, lead, or source identity. Such corrections require a controlled reconciliation job. Timing, labels, and duration may be corrected by increasing revision. The latest revision replaces the earlier contribution rather than adding another attempt.

**DATA-06.** Incomplete/provisional events are stored upstream but SHALL NOT update the model. An unanswered call is a valid completed outcome; absence of a call is not an unanswered outcome.

### 4.3 Seller profile and decision context

Seller profiles contain `seller_id`, nullable `category_group`, `turnover_band`, `business_type`, `effective_from`, and `profile_version`. The adapter owns category/turnover normalization and an explicit `UNKNOWN` value. It SHALL NOT derive segment features from future outcomes. Point-in-time profiles are required for valid historical evaluation; if unavailable, the report states the limitation and omits affected subgroup claims.

Each request context includes seller and lead IDs, lead creation time, lead expiry, context version, suppression/opt-out status, verified attempt count, completed and reserved calls per local day, last call end, and optional requested callback time. Production context comes from trusted adapters; clients cannot override caps, suppression, assignment, or wall clock.

### 4.4 Historical adapter mapping

| Source field named in v1.1 | Canonical field |
|---|---|
| `fk_glusr_usr_id` | `seller_id` |
| `data_hotlead_disposition_dtlid` | Candidate `attempt_id`; verify it represents one actual dial rather than one disposition change. |
| `lead_call_status` | `answered` through a versioned mapping table. |
| `lead_tbro_time` | `requested_callback_at`, only if its documented semantics match. |
| `call_attempt_count` | Candidate `attempt_number`; verify its grouping and reset semantics. |
| `redis_bucket` | `source_bucket` |

**DATA-07.** The source mapping for `lead_id`, call end, finalized time, revisions, and actual-dial identity must be verified before real backfills. Do not fabricate missing values. Historical predictive training may use valid start/label rows without retry metadata, but SHALL mark them ineligible for retry evaluation. If finalized time is missing, run only an explicitly labeled optimistic retrospective replay, not a delayed-feedback acceptance test.

**DATA-08.** Import reports SHALL count input rows, unique attempts, duplicates, corrections, exclusions by reason, unknown mappings, missing profiles, label contradictions, and retained rows. Quarantine contradictory rows without silently coercing their labels.

## 5. Reward and mathematical model

### 5.1 Reward

**MOD-01.** For finalized attempt i:

```text
y_i = w_meeting * meeting_fixed_i
    + w_answered * answered_i
    - c_dial
    - w_not_interested * I(disposition_i = NOT_INTERESTED)
```

Defaults are `(1.0, 0.1, 0.02, 0.0)`. Examples: unanswered = −0.02; answered without meeting = 0.08; meeting = 1.08. This is a utility score, not a probability. Constant cost does not alter the ranking of eligible timestamps. All reward changes create a new model compatibility ID and require rebuilding statistics.

### 5.2 Basis and state

**MOD-02.** Use this exact interleaved ordering:

```text
phi(t) = [1, sin(2*pi*t/24), cos(2*pi*t/24), ...,
             sin(2*pi*K*t/24), cos(2*pi*K*t/24)]
d = 2*K + 1
t = local_hour + local_minute/60 + local_second/3600
```

A scalar input returns shape `(d,)`; an array of n times returns `(n,d)`; an empty array returns `(0,d)`. Use float64. Candidate K values are 2, 3, and 4; development default K=4. The hourly basis contains no weekday, lead-age, or attempt-number effects.

**MOD-03.** Model `y = phiᵀ w_s + error`, with Gaussian working noise variance `sigma2 > 0`, prior `w_s ~ N(mu0, Sigma0)`, precision `Lambda0 = Sigma0⁻¹`, and `eta0 = Lambda0 mu0`. The Gaussian likelihood is an approximation for a discrete, bounded reward. Predictions SHALL NOT be labeled answer or meeting probabilities.

**MOD-04.** Start seller statistics at zero and update each unique finalized attempt exactly once:

```text
A += outer(phi, phi) / sigma2
b += y * phi / sigma2
n += 1
Lambda = Lambda0 + A
L = cholesky(Lambda)              # lower triangular
mu = solve(L.T, solve(L, eta0+b))
```

`gamma` SHALL equal 1 in release 1; reject other values. In this version `n_eff = n`. Addition is order-independent up to floating-point roundoff. Do not claim exact Bayesian inference for the real outcome-generating process; the posterior is exact only for the specified working model and fixed prior.

**MOD-05.** Updating a revision subtracts the previous `A,b` contribution and adds the new one; n stays unchanged. Contributions use actual call-start features. Remove or add eligibility when a corrected call crosses the model history boundary. Periodic reconciliation rebuilds state from the current finalized attempt ledger to bound numerical drift.

### 5.3 Scoring and uncertainty

For candidate feature vector phi, return:

```text
expected_reward = phi.T @ mu
latent_std = sqrt(max(0, dot(solve(L, phi), solve(L, phi))))
predictive_std = sqrt(latent_std**2 + sigma2)
prior_weight = trace(Lambda0) / trace(Lambda0 + A)
```

**MOD-06.** `prior_weight` is a basis-dependent diagnostic, not the fraction of the prediction caused by the prior. Return finite values in `[0,1]` within numerical tolerance. The cold-start value is 1. Do not clip scores to `[0,1]` or use Gaussian intervals as calibrated probability intervals.

**MOD-07.** Failed Cholesky, nonfinite parameters, or incompatible state triggers a named fallback and an alert; never return fabricated uncertainty. Bundle validation SHALL prevent non-positive-definite priors. A live service SHALL NOT silently change the statistical model by repeatedly adding arbitrary jitter.

## 6. Prior training and compatibility

### 6.1 Deterministic first-release estimator

**TRAIN-01.** Segment resolution is `cell → group_turnover → group → global`. A segment is eligible when it has at least 2,000 finalized training attempts and 200 distinct sellers. Thresholds are configurable and versioned. Missing segment dimensions stop descent at the nearest valid parent. The global segment must pass the same gate for a live model; small synthetic fixtures may explicitly override it.

**TRAIN-02.** Define `D = diag(0,1,1,4,4,...,K²,K²)`. Fit the global mean by minimizing:

```text
sum_i (y_i - phi_i.T @ mu)**2
  + lambda_smooth * mu.T @ D @ mu
  + 1e-6 * dot(mu,mu)
```

For each eligible child, add `lambda_parent * ||mu-mu_parent||²`. Fit parents before children. Use linear solves, not explicit inversion. The `1e-6` numerical ridge applies to all coordinates; the harmonic smoothness penalty leaves the intercept unpenalized.

**TRAIN-03.** Release 1 uses a deliberately regularized covariance rather than claiming to estimate full between-seller heterogeneity:

```text
Sigma0 = alpha * diag(1, 1,1, 1/4,1/4, ..., 1/K²,1/K²)
```

Use the same alpha at all levels. This covariance is a tuning choice expressing shrinkage, not an empirical estimate of seller heterogeneity. Full random-effects covariance estimation requires a separate model revision and evidence that it improves held-out performance.

**TRAIN-04.** Search this finite development grid with `gamma=1`:

| Parameter | Candidates |
|---|---|
| K | 2, 3, 4 |
| `lambda_smooth` | 0.1, 1, 10 |
| `lambda_parent` | 10, 100 |
| `alpha` | 0.01, 0.1, 1.0 |

For each K/penalty combination, estimate sigma2 as the mean squared residual of frozen prior-mean predictions on the calibration month, floored at `1e-4`. Estimate before applying that month's observations to seller state. Tune alpha and the other choices by predictive Gaussian negative log-likelihood during the following month, with posterior updates only after labels become available. Report MSE too. Tie within `1e-8` mean loss: choose smaller K, then larger lambda_smooth, then larger lambda_parent, then smaller alpha. A value of 0.25 is permitted only for synthetic/bootstrap runs and must be flagged as uncalibrated.

### 6.2 Supported hours

**TRAIN-05.** Build an explicit set of supported 15-minute local-time bins for each eligible segment from prior-training data, not merely minimum and maximum observed hours. A bin is supported if it contains at least 50 attempts from 30 distinct sellers; thresholds are versioned. Bins are half-open intervals. A query timestamp belongs to its containing bin. An unsupported bin cannot be made supported by interpolation.

If the resolved segment has no usable supported candidates for a request, retry the whole resolution using its parent curve and support. Do not mix a child curve with a parent's support without logging a model change. If global has none, use the approved baseline or return no slot. Randomized exploration also respects this support mask in release 1.

### 6.3 Time boundaries and reproducibility

**TRAIN-06.** For the supplied six-month period, use these disjoint ranges in local time:

| Purpose | Interval, start inclusive and end exclusive |
|---|---|
| Population prior fitting | 2026-04-01 to 2026-07-01 |
| Noise calibration and seller-state warmup | 2026-07-01 to 2026-08-01 |
| Hyperparameter validation | 2026-08-01 to 2026-09-01 |
| Final untouched test | 2026-09-01 to 2026-10-01 |

At each boundary use only outcomes finalized before that instant. Prior-fitting rows SHALL NOT also be replayed into the same seller state in this release. This split sacrifices some training volume but prevents silently treating data reused in fitted priors and likelihood statistics as independent evidence. Final testing uses frozen hyperparameters and a seller-state replay of July–August; it does not retune on September.

**TRAIN-07.** A bundle SHALL include format version, unique bundle ID, model compatibility ID, K, exact feature ordering, sigma2, reward parameters, support masks, segment dictionary, means and precisions, profile-mapping version, data selection boundaries, training statistics, validation metrics, config hash, code revision, and checksums. Use JSON metadata plus non-pickled numeric arrays. Validate dimensions, finite values, symmetry, SPD, and checksums on load.

**TRAIN-08.** Model compatibility covers feature definition, reward, sigma2, state-history start, and normalization. Compatible prior-mean/covariance refreshes may reuse statistics only when prior-training data remain disjoint from the state history. Changing the prior cutoff, K, reward, or sigma2 requires a new state namespace and deterministic backfill. Atomically activate a matching bundle/state namespace pair. Retain the previous pair for rollback; never relabel old A/b as compatible.

**TRAIN-09.** Segment keys use canonical JSON arrays and a bundle dictionary with collision checks, not language-specific `hash()`. Profile changes resolve a new prior at request time; history remains seller-level. Record the chosen profile version and segment on every decision.

## 7. Recommendation policy and temporal constraints

### 7.1 Eligibility

**POL-01.** Production uses server time. A clock may be injected only in tests and offline replay. All returned scheduled timestamps must be strictly in the future relative to the captured server time.

The caller supplies `earliest_at`, `latest_at`, and lead identity. The service intersects these with lead expiry, approved business calendar, suppression, remaining caps, and policy-specific not-before constraints. For a fresh lead, also apply `lead_sent_time + max_initial_delay_minutes`. This maximum is a required production setting; if already breached, use the approved urgency fallback rather than resetting the delay clock.

**POL-02.** Calendar windows are start-inclusive and end-exclusive. Development configuration is 08:00–18:00 Asia/Kolkata, Monday–Saturday, no holidays; production SHALL provide an explicit versioned calendar. An exact 18:00 timestamp is not permitted by that development window. Calculate next working day through the calendar, not by adding 24 hours.

**POL-03.** Generate quarter-hour timestamps between eligibility bounds on each permitted date, with at most seven calendar days of horizon. If the request exceeds seven days, return validation error. In each permissible interval, also include its earliest feasible timestamp if no quarter-hour grid point fits before its deadline; deduplicate exact timestamps and check support. This ensures an urgent eligible interval shorter than 15 minutes can still produce one action.

Use a development dispatch lead time of five seconds when calculating earliest feasibility; production scheduler lead time is required configuration. Explicit callbacks use their requested timestamp, not grid rounding.

**POL-04.** If a day has no remaining call capacity, remove that day. If no candidates remain, return `NO_ELIGIBLE_SLOT` with null timestamps; never relax a hard constraint to return a recommendation. The actual scheduler must recheck capacity atomically.

### 7.2 Selection and secondary peak

**POL-05.** Deterministic treatment selects the maximum posterior expected reward over eligible candidates. Ties within `1e-12` choose the earliest timestamp. Zero-state sellers use mode `PRIOR_ONLY`; others use `EXPLOIT`. Choosing a time does not imply evidence of individual personalization; log n and prior weight.

**POL-06.** Exploration selects uniformly over the exact same final candidate set. Persist the selected action and its probability `1 / candidate_count`. No posterior sampling or post-sampling softmax is permitted in release 1.

**POL-07.** The optional secondary peak is descriptive, not another randomized action. Compute local maxima of the expected curve on contiguous supported quarter-hour runs for the selected date. A plateau is one peak represented by its earliest point; an endpoint can be a peak if strictly better than its one neighbor; a wholly flat run has no peak. Choose the highest local maximum at least two hours from the selected primary timestamp, breaking ties earlier. Return null when none exists. Never fall back to an arbitrary point and label it a peak.

For next-working-day retry selection, recompute primary and secondary peaks on that date's eligible candidates. Do not carry a previous date's timestamp into a new day.

### 7.3 Baseline and fallback

**POL-08.** `BaselinePolicy` is an injected adapter returning an approved timestamp and reason. The existing baseline must be documented and versioned; do not assume every baseline call follows a 15-minute rule. Development stub: earliest feasible timestamp for first calls; +15 minutes for a first unanswered retry. The stub SHALL fail production startup checks.

Fallback order is: compatible seller model → resolved segment prior when the seller is verified new → compatible cached prior policy when explicitly allowed → approved baseline → no slot/unavailable. A cache miss is not proof of zero history. Numerical corruption, unknown context, or stale restrictions cannot be converted into an unrestricted cold start.

## 8. Retry rules and scheduler protocol

### 8.1 Precedence

**RET-01.** Apply these checks in order: suppression/opt-out or inactive lead; terminal disposition; lead expiry; total-attempt limit; seller daily cap/reservations; explicit callback; disposition rule; calendar/support/deadline projection. The daily cap excludes a date and may permit a later one; the lead-attempt cap permanently stops automatic retries for that lead. Terminal Meeting Fixed, Not Interested, and General produce no automatic retry in release 1. Unknown disposition produces manual review.

**RET-02.** Retry intervals are anchored to the actual `call_end_time`. Let `base = max(call_end_time, server_now)` only for preventing past actions; do not restart nominal delay calculations at event delivery. All rules additionally enforce the configured minimum inter-call gap and dispatch lead time. A callback before this earliest feasible instant is stale and requires review.

### 8.2 Dynamic retry decision table

This table applies only in the separately enabled dynamic-retry treatment. During the timing experiment, every arm uses the same approved baseline retry rule.

| Latest finalized outcome | Conditions | Proposed action |
|---|---|---|
| Any | Suppressed, terminal lead, expired, or lead-attempt cap reached | `STOP`, no timestamp. |
| Meeting Fixed / Not Interested / General | Otherwise | `STOP`, no automatic retry. |
| Unknown | Otherwise | `MANUAL_REVIEW`, no timestamp. |
| Call Later / Busy | Explicit callback in the future and permissible | Exactly that callback timestamp. |
| Call Later / Busy | Callback stale, outside approved calendar, beyond expiry, or conflicts with hard constraints | `MANUAL_REVIEW`; preserve requested time in reason metadata. Do not silently move a seller's explicit appointment. |
| Call Later / Busy | No callback | Search expected-curve peaks in `[call_end+60m, call_end+240m]` intersected with future eligible times. Select highest score, then earliest. If none, propose `call_end+120m`. |
| Not Answered | First attempt; double-call disabled | Propose `call_end+15m` through the common constraint projection. This value is a development baseline assumption until approved. |
| Not Answered | First attempt; separately approved double-call arm enabled | Draw uniformly from integer delays `{120,121,...,300}` seconds, once per retry decision. Persist delay and probability `1/181`; then enforce constraints. |
| Not Answered | Attempt number ≥ 2 | Next eligible working day after the actual call's local date, at secondary peak if present, otherwise primary. If that day is already past when processed, use the first future eligible working date. |

**RET-03.** For non-callback proposals only: if the proposed instant is stale, move to the first feasible instant at or after server_now plus dispatch lead time and minimum gap. If outside a work window or daily capacity is exhausted, move to the first eligible later date and select its primary peak, or earliest supported slot if no peak. Apply expiry and request horizon after every projection. If no permissible action exists, return no slot. Recheck temporal bounds; never move backward to a passed peak.

An in-window non-callback proposal that lies in an unsupported bin moves to the earliest supported feasible candidate at or after that proposal. If none remains that date, apply the next-eligible-date rule. Explicit callbacks are business instructions and may occur outside model support if all hard calling constraints are satisfied; report null model diagnostics and exclude them from time-policy OPE. Model support is not a reason to move an otherwise permissible explicit appointment.

Double-call randomness is separate from time-policy exploration. Record the draw before projection, the projected action, and an OPE-exclusion reason. The simple `1/181` delay probability SHALL NOT be reported as the probability of a projected final timestamp when multiple draws map to the same action.

**RET-04.** A retry decision is uniquely keyed by `(source, attempt_id, revision, retry_policy_version)`. Repeated requests return the same persisted result. A newer revision supersedes pending work for an older revision. If a newer actual attempt exists for that seller/lead, return `SUPERSEDED` rather than generating a stale retry.

### 8.3 Scheduler integration

**RET-05.** Scheduler commands contain `decision_id`, `seller_id`, `lead_id`, `scheduled_at`, `expires_at`, source attempt revision, context version, and policy version. Submission is idempotent by decision ID. The scheduler SHALL return `RESERVED`, `REJECTED_CONSTRAINT`, `REJECTED_CAPACITY`, or `ALREADY_EXISTS` plus reservation ID when applicable.

**RET-06.** Before reservation and again before dispatch, the scheduler SHALL check current suppression, lead status, expiry, latest outcome, total attempts, daily completed calls plus active reservations, and per-seller minimum spacing across leads. Reserve at most one pending call per seller/lead. Release reservations on cancellation or expiry. Reservation conflict requests a new decision with a new decision ID; do not silently move an action and retain its original propensity.

**RET-07.** Emit an execution record for actual starts, cancellations, rejections, and missed windows. Include planned timestamp, actual timestamp, reasons, and decision ID. Model updates use actual starts; experiment reporting retains assignment even when execution deviates.

## 9. Persistence, deduplication, and recovery

### 9.1 Authoritative tables

| Table | Required keys and contents |
|---|---|
| `attempt_latest` | PK `(source, attempt_id)`; seller/lead identity, revision, canonical payload hash, finalized labels and times. |
| `attempt_revisions` | Append-only audit of accepted revisions with ingestion time and payload. |
| `seller_model_state` | PK `(model_compatibility_id, seller_id)`; A, b, n, state_version, max call time, last commit time. |
| `decisions` | Unique decision/request ID; canonical request hash, response, full candidate set/probabilities, versions, state watermark. |
| `retry_decisions` | Unique RET-04 key; result, supersession status, scheduler linkage. |
| `outbox` | Transactional events for committed state changes, decisions, and invalidations. |

**STATE-01.** Within one database transaction: establish the attempt revision lock, lock/create the seller-state row, compare revision and payload, calculate the delta, write latest and audit rows, update statistics and state version, and append an outbox record. Commit before acknowledging success. Use a consistent lock order. Duplicate same revision/same payload is a successful no-op; same revision/different payload is `409 CONFLICT`; an older revision is a successful `STALE_REVISION` no-op.

**STATE-02.** If the process crashes before commit, no contribution is visible. If it crashes after commit but before acknowledgement, redelivery is a no-op. No seven-day expiring set is the correctness boundary. Queue adapters acknowledge only after database commit; exhausted transient retries go to a dead-letter queue, never become silent success.

**STATE-03.** Per-seller concurrent updates SHALL serialize without lost increments. Persist state_version as signed 64-bit monotonic integer; never a 16-bit CAS value. Database outages fail writes with retryable 503. Retry transactions at most three times with bounded jitter, then fail visibly.

**STATE-04.** A model state is eligible only for finalized attempts whose call start is in its configured state-history interval. Older events stay in the audit ledger but do not alter that namespace. Revisions crossing the interval boundary adjust n accordingly. A frozen namespace records its source ingestion watermark.

**STATE-05.** Redis cache entries include compatibility ID, state_version, and cached-at time. Outbox application only replaces entries with newer state versions; out-of-order messages cannot revert state. Maximum acceptable state staleness is five seconds by development default, production-configured. On miss or excess staleness read PostgreSQL. Unknown restriction freshness fails to baseline/no slot; model-cache freshness does not establish calling eligibility.

**STATE-06.** Cold-start zero state is allowed only after an authoritative not-found result. Do not expire authoritative states silently. Retention, compaction, and rebuild have explicit watermarks and a documented replay horizon. Production activation requires an approved retention period and ability to reject or reconcile events older than retained dedupe history.

**STATE-08.** Backfill an inactive namespace from a consistent database snapshot with an ingestion watermark. Maintain an attempt-revision application ledger for that namespace, replay subsequent durable change records in commit order, and apply revisions idempotently using their old/new contributions. For activation, briefly pause outcome processing, drain and reconcile through the final committed watermark, atomically switch the bundle/namespace pointer, then resume processing. Queue or retry incoming writes during the pause. Keep the previous namespace current through the same durable changes while it is a rollback target. Never activate a snapshot that omits changes committed during its build.

### 9.2 Compact cache codec

The ≤1 KiB target applies to the serialized numerical state, not the database ledger or complete system.

**STATE-07.** Cache codec uses little endian, packed upper triangle in row-major `(i,j), j≥i`, followed by b, all float64. A 48-byte header contains magic u32, format u16, d u16, compatibility tag u64, n u64, state_version u64, last-commit epoch milliseconds i64, and CRC32 u32 plus reserved u32. Check the full compatibility ID in the cache key/metadata; the tag is not the sole collision protection. CRC covers the header with the CRC field zeroed plus payload. Reserved bits must be zero. Signed timestamps and unsigned counters must be range-checked.

```text
bytes(d) = 48 + 8 * (d*(d+1)/2 + d)
bytes(9) = 480
150,000 numeric payloads = 72,000,000 bytes, before storage overhead
```

Reject wrong length, checksum, schema, dimension, or nonfinite payload. Preserve exact float64 bit patterns on a valid round trip. Raw call history, decisions, and retry reservations are budgeted separately.

## 10. Service contracts

### 10.1 API conventions

**API-01.** All mutating decision requests carry a UUID `request_id` supplied by the trusted caller. Repeating an ID and identical canonical request returns its persisted response; repeating it with a different payload returns 409. Replaying a stale recommendation does not authorize dispatch: the scheduler still applies RET-06. A deliberate reschedule uses a new request ID.

Concurrent identical requests SHALL return the single winning committed decision under the unique request ID. A losing transaction discards its independently computed draw and reads the committed winner; it must never return its uncommitted candidate. Canonical request hashing uses sorted-key UTF-8 JSON after timestamp normalization to UTC. Optional fields are normalized to explicit null before hashing.

Endpoints use authenticated service identity; authorization restricts outcome producers, decision callers, and model administrators separately. TLS and secret-management integration are production requirements. Health endpoints SHALL NOT expose seller data or credentials.

| Endpoint | Request | Response |
|---|---|---|
| `POST /v1/best-time` | request_id, seller_id, lead_id, earliest_at, latest_at | 200 persisted recommendation or no-slot decision. |
| `POST /v1/outcomes` | Canonical finalized outcome from §4 | 200 `APPLIED`, `DUPLICATE`, or `STALE_REVISION`, with state_version. Synchronous commit in release 1. |
| `POST /v1/retries` | request_id, source, attempt_id, expected_revision | 200 persisted retry result; reads the authoritative outcome rather than accepting duplicate labels. |
| `GET /healthz` | None | Process liveness. |
| `GET /readyz` | None | Readiness of valid model/config and required stores/adapters. |
| `GET /metrics` | Authorized internal caller | Operational metrics without high-cardinality seller labels. |

**API-02.** Errors: 401/403 authentication/authorization, 404 unknown lead/attempt, 409 request or revision conflict, 422 schema or interval violation, 429 throttling, 503 required dependency unavailable. Include `error_code`, `message`, `request_id`, and `retryable`; never include internal stack traces. Failed model scoring may return a 200 baseline result only when fallback is explicitly permitted and identified.

### 10.2 Recommendation response

Required fields: `decision_id`, `request_id`, `seller_id`, `lead_id`, `status`, `scheduled_at`, `secondary_at`, `reason_code`, `mode`, `assignment`, `experiment_id`, `policy_version`, `bundle_id`, `model_compatibility_id`, `profile_version`, `calendar_version`, `context_version`, `state_version`, `n_attempts`, `n_eff`, `prior_level`, `prior_weight`, `expected_reward`, `latent_std`, `predictive_std`, `candidate_count`, `action_probability`, `assignment_probability`, `ope_eligible`, `created_at`, and `valid_until`.

Enums: status is `RECOMMENDED`, `NO_ELIGIBLE_SLOT`, `STOP`, `MANUAL_REVIEW`, or `SUPERSEDED`; mode is `EXPLOIT`, `PRIOR_ONLY`, `UNIFORM_EXPLORE`, `BASELINE`, or `NONE`. Non-model responses have null model diagnostics. Null is not zero. Expected reward is for the chosen action under the posterior mean, including when that action was randomized.

Assignment is `CONTROL`, `TREATMENT`, `EXPLORE`, or `SHADOW`. Shadow proposals use `SHADOW` with null experiment ID and probabilities and `ope_eligible=false`. When an experiment is disabled in live mode, return approved baseline behavior as `CONTROL` with null experiment ID; no implicit treatment activation is allowed. With no selected action, action_probability is null and candidate_count is zero. Unknown baseline probabilities remain null. No-slot/stop/review/superseded responses have null scheduled_at, secondary_at, and valid_until.

`valid_until` is the earliest of lead expiry, context expiry, and scheduled_at plus the approved execution tolerance; no timestamp when no action exists. Default execution tolerance for synthetic tests is five minutes, but dispatch still must satisfy the calling window. `created_at` and `valid_until` are timestamps, not fractional hours.

Example partial response for a two-candidate exploration decision:

```json
{
  "status": "RECOMMENDED",
  "scheduled_at": "2026-10-09T11:15:00+05:30",
  "secondary_at": null,
  "mode": "UNIFORM_EXPLORE",
  "assignment": "EXPLORE",
  "candidate_count": 2,
  "action_probability": 0.5,
  "assignment_probability": 0.05,
  "ope_eligible": true
}
```

The example is illustrative, not a complete schema instance. OpenAPI and JSON Schemas generated from the implementation SHALL include all required fields and enum values above.

### 10.3 Core interfaces

```python
fourier(times, k) -> ndarray
reward(outcome, reward_config) -> float
apply_contribution(state, old_outcome, new_outcome, model_config) -> SellerState
posterior(state, prior) -> Posterior
resolve_prior(profile, bundle, candidate_intervals) -> ResolvedPrior
generate_candidates(context, calendar, support, clock) -> list[datetime]
select_action(candidates, posterior, assignment, rng) -> PolicyDecision
plan_retry(latest_outcome, context, config, calendar, clock) -> RetryPlan
record_outcome(outcome) -> CommitResult
recommend(request) -> Recommendation
```

The model, candidate generation, and retry planning functions SHALL be pure and receive their clock/RNG explicitly. Network and database access belong to adapters. Do not pass a process-global mutable random generator between concurrent requests.

## 11. Experiment assignments and decision logging

**EXP-01.** An experiment is disabled by default. When enabled, assign sellers, not individual calls, to control 45%, treatment 50%, explore 5%. Compute SHA-256 over canonical UTF-8 JSON `[experiment_id, salt, seller_id]`; interpret the first eight bytes as unsigned big-endian integer and divide by 2^64. Intervals are control `[0,.45)`, treatment `[.45,.95)`, explore `[.95,1)`. Fractions and salt are immutable within an experiment. Changing them requires a new experiment ID.

**EXP-02.** Distinguish assignment probability from the conditional probability of selecting an action. On an exploration-arm decision with m candidates, the latter is 1/m. Deterministic treatment has probability 1 for its chosen action within its arm, and zero elsewhere. Do not multiply by the arm fraction unless the estimator explicitly models the full mixture. A legacy baseline with unknown stochastic behavior has null action probability and is not OPE-eligible.

**EXP-03.** Persist the complete post-constraint candidate timestamps in sorted order, selected index, action probabilities, context snapshot/hash, state and model versions, assignment, seed/PRNG algorithm version, and every fallback before returning a successful actionable decision. Repeated request IDs reuse this record, not a new random draw. Failing durable decision logging fails the actionable request with 503.

**EXP-04.** A baseline fallback, external reschedule, missing final outcome, override, or unsupported action is logged with an explicit evaluation status. Missing outcomes are not automatically coded as zero reward. Report these cases and their rates by arm. Exclusion from OPE does not remove a seller or lead from intention-to-treat experiment reporting.

**EXP-05.** For the initial timing experiment, restrict primary policy comparison to the first eligible scheduling decision per seller/lead. Keep retry rules fixed. Full-sequence benefits and double-call policies require a separate seller-randomized online experiment; one-step bandit OPE is not a valid estimator of their total long-term effect.

## 12. Evaluation and business measures

### 12.1 Phase 0 report

**EVAL-01.** Produce distributions of observations per seller, hour-bin support, weekday, segments, call status, and lead age. Report meeting and answer rates with denominators, actual call identity checks, event delays, duplicate/revision frequency, and missing data. Analyze gap/outcome associations within seller–lead sequences as descriptive evidence, not causal retry effects. No dataset-derived figures may be fabricated when data is unavailable.

Compare constant global, segment-only, and personalized models. Weak personalization evidence is a reason to ship segment-only behavior, not to manufacture individual peaks.

### 12.2 Predictive backtest

**EVAL-02.** In chronological replay, process each recommendation at its decision time using only labels finalized by that time. If a label becomes available at exactly the same instant, process the recommendation first, then the label. Evaluate the prediction at the logged actual call time; do not attach that outcome to the model's preferred unobserved time.

Freeze prior and selected hyperparameters for the September test. Metrics: reward MSE, Gaussian predictive negative log-likelihood, expected-score versus observed mean by supported time bin, and results by history-count groups 0, 1–4, 5–9, ≥10. These groups are descriptive, not a guarantee of sufficient evidence. Report uncertainty coverage as an empirical diagnostic. An overall reduction in prediction loss is not proof of policy lift.

### 12.3 One-step OPE

**EVAL-03.** Prefer exploration-arm records with exact logged probabilities and common eligible action definitions. Historical estimated-propensity OPE SHALL be labeled assumption-dependent and is not a launch gate proving lift. Do not assign a fabricated positive probability to an action the historical policy never took.

For N eligible records, logged behavior probability p_i, target probability q_i, reward r_i, and cross-fitted reward estimates qhat(x,a):

```text
w_i = q_i / p_i
IPS = mean(w_i * r_i)
SNIPS = sum(w_i * r_i) / sum(w_i)
DR = mean(sum_a pi(a|x_i)*qhat(x_i,a)
          + w_i*(r_i-qhat(x_i,a_i)))
ESS = sum(w_i)**2 / sum(w_i**2)
```

The q_i probability notation and qhat reward model are distinct. Fit qhat only on separate seller folds or earlier data; never on the same evaluation outcomes. Evaluate a frozen target policy, not an adaptive learner with counterfactual state updates that are unavailable in the logs.

**EVAL-04.** Report untrimmed estimates and a sensitivity estimate with weights clipped at 10, explicitly noting clipping changes bias. If sum weights is zero, return `NOT_ESTIMABLE`. Report N, missing records, action overlap, maximum weight, ESS, clipping fraction, and 95% confidence intervals from 1,000 seller-cluster bootstrap replicates with a fixed seed. An ESS below 1,000 or 10% of N is a development warning, not a universal statistical theorem or substitute for power analysis.

Planned slots and actual dispatches must remain distinguishable. OPE estimates the value of the logged scheduling action with its downstream execution mechanism; evaluating exact actual dial-time effects from planned-time propensities is invalid without additional assumptions. Report execution deviations and avoid causal claims from selectively retaining only successfully executed recommendations.

### 12.4 Online experiment metrics

**EVAL-05.** Primary intention-to-treat metric is the fraction of eligible seller–lead pairs with at least one meeting fixed within an approved attribution window from first eligibility, regardless of dispatch or fallback. Development attribution window is seven days; production must approve it. Deduplicate repeated meeting dispositions for the same seller/lead. Wait for window maturity before final analysis.

| Metric | Exact denominator |
|---|---|
| Attempt meeting rate | Finalized actual attempts with a meeting / all finalized actual attempts. Secondary diagnostic; retry count can alter it. |
| Answer rate | Answered actual attempts / finalized actual attempts. |
| Call yield | Unique seller–lead meetings / total actual call duration in hours; report missing-duration rate and suppress if denominator is zero. |
| Retry conversion | Eligible retried seller–lead pairs with a meeting within attribution window / eligible retried pairs; descriptive unless randomized. |
| Not Interested rate | Such finalized outcomes / finalized actual attempts. |
| Delay | Actual first call start minus lead_sent_time, with p50/p95 and SLA breach rate. |
| Call pressure | Actual attempts and reservations per seller/local day; cap violations are counted separately. |

**EVAL-06.** Use seller-cluster confidence intervals for online comparisons because assignment is by seller. Freeze sample size, minimum detectable effect, attribution window, guardrail thresholds, experiment duration, and stopping rule before exposure. The source targets of +15% meeting rate, +8% answer rate, and +20% call yield remain business hypotheses; passing software acceptance SHALL NOT claim these gains.

## 13. Nonfunctional requirements and operational behavior

| ID | Requirement and measurement boundary |
|---|---|
| NFR-01 | Numerical seller-state payload ≤1 KiB at K≤4. Benchmark actual codec bytes. |
| NFR-02 | Target optional Redis numerical-state cache ≤200 MB for 150,000 sellers, measured allocated memory including keys/metadata. Report baseline subtraction and fragmentation separately. Excludes ledger, decisions, profiles, priors, and replicas. |
| NFR-03 | Preserve original target `recommend` p99 <10 ms, measured from service receipt through durable decision logging and response serialization, with required profile/context access included. Not assumed satisfied. |
| NFR-04 | Preserve original target finalized-outcome processing p99 <2 ms from service receipt through durable transaction commit. Separate CPU-only and full-service results; never substitute one for the other. |
| NFR-05 | Bench on declared hardware: development reference 4 vCPU/8 GiB service, separate 4 vCPU/8 GiB database, same-zone networking; optional dedicated Redis. Run 5-minute warmup then 15-minute measured load at 500 recommendations/s plus 100 outcomes/s, 150k seeded sellers, 10% cold reads, and a 10% hot-seller traffic group. Production load profile must replace these assumptions before launch. |
| NFR-06 | Report p50/p95/p99, errors, throughput, conflicts, queue lag, cache misses, CPU, RSS, DB size, and network latency. A failed target blocks the performance gate until optimization or an explicit SRS/SLO revision; agents cannot quietly weaken it. |
| NFR-07 | No lost/duplicate model contributions under tested crash/redelivery scenarios. Durable accepted outcomes must survive process restart. Database disaster recovery and retention RPO/RTO are deployment-owned release inputs. |
| NFR-08 | Emit reason-coded counts for no-slot, fallback, corruption, conflicts, stale revisions, invalid data, dispatch deviation, and cap rejection. Provide structured logs with request/decision IDs and versions; keep seller IDs out of metric labels. |
| NFR-09 | Traceable model/config/code versions on decisions, bundles, and state transactions. UTC log times plus explicit business timezone. Reproducible seeded offline runs. |
| NFR-10 | Authenticated and authorized API; managed secrets; protected seller-level logs; retention/deletion policy; no raw telephone numbers or call transcripts required by this service. |
| NFR-11 | Kill switch returns control/baseline behavior when permissible, disables exploration and new dynamic retries, and requests cancellation of pending experimental reservations. It does not remove suppression or caps. |

The 10 ms/2 ms targets may require revisiting after measurement because the durable correctness requirements add work absent from v1.1's estimate. This is an explicit feasibility risk, not a completed benchmark.

## 14. Required configuration

**CFG-01.** Settings are validated on startup and have a content hash. Reject invalid fractions, negative weights/costs, unsupported gamma, zero sigma2, inconsistent windows, mismatched model dimensions, and unknown keys. Secrets are environment-injected, not committed YAML.

| Setting | Build/simulation value | Production requirement |
|---|---|---|
| runtime mode | shadow | Explicit activation mode. |
| timezone | Asia/Kolkata | Same unless new model/calendar version. |
| calendar | Mon–Sat, 08:00–18:00 | Approved calendar and holidays. |
| initial delay maximum | 15 minutes | Business-approved value; no unbounded default. |
| expiry when upstream absent | None | Missing expiry prevents actionable recommendations. |
| max calls/seller/day | 3 | Approved value enforced externally and locally. |
| max attempts/seller/lead | 5 | Approved value including non-BTC calls. |
| minimum inter-call gap | 15 minutes | Approved value; separate approved override for double-call experiment. |
| dispatch lead time | 5 seconds | Measured scheduler requirement. |
| decision execution tolerance | 5 minutes | Approved dispatch policy. |
| cache staleness | 5 seconds | Approved freshness budget. |
| experiment | disabled | Immutable approved experiment configuration. |
| double-call | false | Explicit separate activation and compatible minimum gap. |
| terminal retry dispositions | Meeting Fixed, Not Interested, General | Confirm rule table before live activation. |
| retention and replay horizon | Local fixtures only | Explicit approved values and deletion workflow. |

## 15. Verification and acceptance matrix

Tests SHALL verify observable behavior and failure modes, not only mirror implementation code. Numeric comparisons use `atol=1e-9, rtol=1e-9` on well-conditioned float64 fixtures unless stated otherwise.

| Test ID | Required scenario and pass condition | Requirements |
|---|---|---|
| T01 | Scalar, array, empty-array basis shapes; phi(0); periodicity over 24 h; exact ordering. | MOD-02 |
| T02 | Rewards −0.02, 0.08, 1.08; contradictory labels rejected. | DATA-04, MOD-01 |
| T03 | Incremental statistics/posterior equal batch solution; shuffled events agree with gamma=1; gamma=.99 rejected. | MOD-04 |
| T04 | Posterior SPD and uncertainty match a trusted small-matrix solution; cold-start prior weight 1. | MOD-06, MOD-07 |
| T05 | Parent backoff, missing profiles, unsupported-bin holes, and sparse segment gates. | TRAIN-01, TRAIN-05 |
| T06 | Repeated training is reproducible; September labels cannot alter bundle selection; future profiles excluded. | TRAIN-04, TRAIN-06 |
| T07 | K/reward/sigma2/history-boundary mismatch blocks reuse; rollback restores matching state/bundle pair. | TRAIN-08 |
| T07b | Outcomes and corrections committed during backfill are included before activation; rollback namespace remains current. | STATE-08 |
| T08 | Day-end, holiday, end-exclusive 18:00, all-past intervals, short urgent interval, seven-day limit, and no-slot cases. | POL-01–04 |
| T09 | Equal-score earliest tie; flat/unimodal/bimodal curves; secondary null when no separated local maximum. | POL-05, POL-07 |
| T10 | Every retry table row; stale callback review; delayed delivery; caps across multiple leads; newer attempt supersedes old retry. | RET-01–04 |
| T11 | 100 concurrent unique attempts for one seller produce n=100; 100 duplicate deliveries of one attempt produce n=1. | STATE-01–03 |
| T12 | Crash before commit versus after commit before acknowledgement; replay yields one contribution and one logical outbox event. | STATE-02 |
| T13 | Same revision/different payload conflicts; corrected label/time replaces contribution; older revision no-op; boundary-crossing revision adjusts n. | DATA-05, MOD-05, STATE-04 |
| T14 | Codec exactly 480 bytes at d=9; exact valid round trip; corrupt CRC, wrong version, and NaN rejected. | STATE-07 |
| T15 | Out-of-order cache events cannot revert state; stale/missing cache reads source; store outage never invents zero history. | STATE-05–06 |
| T16 | Stable cross-process assignment golden vectors; uniform probabilities sum to 1; 100k draws over a fixed 4-action fixture each within 1% absolute of 25%. | EXP-01–03 |
| T17 | Request replay returns same timestamp/draw; changed payload conflicts; logging failure returns 503; no schedule write in shadow mode. | API-01, EXP-03 |
| T18 | Probability examples: m=2 → .5 conditional action probability, separate .05 assignment; projected double-call is OPE-excluded. | EXP-02, RET-03 |
| T19 | Delayed outcomes unavailable to earlier decisions; prediction scored at actual time, not preferred time. | EVAL-02 |
| T20 | Analytic synthetic OPE fixture verifies IPS/SNIPS/DR; zero support gives not-estimable; clustered uncertainty reproducible. | EVAL-03–04 |
| T21 | Scheduler mock rejects concurrent cap overbooking and duplicate reservations; suppression change cancels pending call. | RET-05–07 |
| T22 | Load report includes complete service boundaries and actual memory; no fabricated latency claims. | NFR-01–06 |
| T23 | Unauthorized operations fail; secrets absent from logs; kill switch disables experimental scheduling. | API-01, NFR-10–11 |

**Software acceptance:** All mandatory behavior tests pass, generated API schemas match §10, deterministic integration demo works, no unresolved severity-1/2 correctness defects, and all benchmark results are recorded. Code can be accepted for shadow testing while business lift is unknown. Live acceptance additionally requires performance gates or an approved target revision and §18 release inputs.

## 16. Repository and agent work packages

### 16.1 Required deliverables

```text
best_time_to_call/
  pyproject.toml
  dependency lockfile
  README.md
  AGENTS.md
  configs/development.yaml
  configs/production.example.yaml
  src/btc/
    config.py  schemas.py  cli.py
    data/          # adapters, normalization, point-in-time joins, splits
    features/      # Fourier and segment resolution
    model/         # reward, stats, posterior, priors, policy
    store/         # SQL ledger/state, transactions, optional Redis cache, codec
    retry/         # calendar, rule table, scheduler protocol
    experiment/    # assignment, logging, evaluation eligibility
    service/       # API, authentication hooks, adapters, outbox worker
    evaluation/    # phase 0, replay, OPE, metrics
  migrations/
  tests/unit/
  tests/integration/
  tests/failure_injection/
  tests/bench/
  fixtures/synthetic/
  artifacts/example_bundle/
  docs/decisions.md
  docs/runbook.md
  docs/traceability.md
```

### 16.2 Work allocation and dependencies

These are tasks for future coding agents. They do not imply that implementation or live deployment has already happened. One integration owner owns shared schemas, dependency versions, migrations, and acceptance decisions.

| Package | Owner role | Inputs and output | Depends on |
|---|---|---|---|
| WP0 | Contract/integration agent | Freeze Pydantic/JSON schemas, enums, interfaces, config validation, fixtures, requirement map. | This SRS. |
| WP1 | Model agent | Pure Fourier/reward/posterior/peak code and T01–04/T09. | WP0. |
| WP2 | Data/training agent | Source adapter, report, deterministic trainer, bundle validator, leakage tests. | WP0; uses WP1 math. |
| WP3 | Persistence agent | Migrations, revision transaction, outbox, cache codec, concurrency/crash tests. | WP0; uses WP1 contribution functions. |
| WP4 | Scheduling agent | Calendar, candidate generation, retry rules, scheduler mock, constraint tests. | WP0; uses WP1 scoring. |
| WP5 | API/experiment agent | Service composition, durable idempotent decisions, assignment/logging, auth hooks. | WP1–4 contracts and implementations. |
| WP6 | Evaluation/QA agent | Replay, OPE, end-to-end test suite, load report, runbook and traceability. | WP2–5. |

After WP0, independent packages may be assigned concurrently with separate file ownership. Agents SHALL NOT independently redefine shared enums, change storage semantics, or choose new business defaults. Interface changes go through the integration owner and update contract tests first.

### 16.3 Agent completion contract

Every work package SHALL provide code, meaningful tests, a short summary of changed behavior, executed checks and results, requirement IDs covered, and remaining limitations. Mark unavailable production adapters explicitly. Never implement a fake connector that returns fabricated successful production results.

Required CLI commands: `btc phase0-report`, `btc train-priors`, `btc validate-bundle`, `btc backfill-states`, `btc backtest`, `btc evaluate-ope`, `btc bench`, and `btc reconcile-state`. Each has `--help`, explicit config/bundle paths, nonzero exit on failure, structured output, and a dry-run option for state mutations. Backfills are idempotent and target a named inactive namespace before activation.

## 17. Coding-agent kickoff instruction

Use this block as the initial instruction for the implementation owner:

> Implement Best Time to Call and Dynamic Retry Engine SRS 2.0 in this repository. Treat SHALL requirements as binding and development defaults as simulation-only. First complete WP0: schemas, interfaces, configuration checks, synthetic fixtures, dependency lockfile, and a traceability map from requirements to tests. Then implement WP1–WP6 in dependency order. Use the exact mathematical, timestamp, event revision, retry, and propensity semantics specified here. Keep the default mode shadow and use injectable clock, RNG, data, baseline, context, and scheduler adapters. Run unit, integration, failure-injection, and declared performance tests; report actual results. Do not place calls, activate experiments, publish a model to production, fabricate datasets, or silently relax acceptance thresholds. Production integration details absent from §18 must remain explicit adapters/configuration gaps. Record any proposed SRS change in docs/decisions.md before relying on it. Finish with a reproducible local demo and a list of production activation requirements still unmet.

## 18. Business and infrastructure release inputs

These do not block building the specified local system. They block live activation because no agent should invent organizational policy or upstream semantics.

| Input | Owner | Required evidence |
|---|---|---|
| Actual dial identity, lead identity, source timezone, call end, finalized/revision semantics | Data engineering / telephony | Field mapping, sample rows, uniqueness and correction tests. |
| Baseline behavior and dispatch contract | Telephony engineering | Versioned baseline adapter, capacity/reservation behavior, execution logging. |
| Calling calendar, suppression and frequency limits | Business policy owner | Versioned approved calendar/caps and authoritative lookup. This document does not determine legal calling rules. |
| Fresh-lead delay and lead expiry | Product owner | Maximum delay, expiry semantics, and urgency fallback. |
| Busy status and terminal dispositions | Operations owner | Approved mapping; whether General can ever require manual follow-up. |
| Not Interested penalty | Product/data science | Keep w=0 initially or approve new reward version and rebuild. |
| Attribution, experiment size and guardrails | Product/data science | Preregistered metrics, maturity window, power calculation, stopping rules. |
| Double-call activation | Operations/product owner | Separate experiment, approved gap override and stop criteria. |
| Infrastructure workload and recovery | Platform owner | Traffic profile, topology, latency benchmark, retention, RPO/RTO and rollback drill. |

No claimed uplift, legal compliance, measured memory footprint, or measured latency is implied by approval of the architecture alone.

## 19. References and change summary

Primary input: **Best Time to Call & Dynamic Retry Engine: Implementation Requirements v1.1**, supplied with this review. Its reported data totals and business targets are treated as unverified inputs until Phase 0.

Technical references used for the review:

1. Dudík, Langford, and Li, [Doubly Robust Policy Evaluation and Learning, ICML 2011](https://icml.cc/2011/papers/554_icmlpaper.pdf). Supports the use of reward-model and propensity-based evaluation; it does not establish that this call dataset meets identification assumptions.
2. Redis, [Transactions documentation](https://redis.io/docs/latest/develop/using-commands/transactions/). Describes transaction and optimistic-locking mechanisms; a separate deduplication write and state mutation do not automatically form one transaction.

Version 2 fixes time and event contracts, separates engineering acceptance from business hypotheses, makes state updates durable and correction-aware, defines compatibility and training boundaries, replaces ambiguous first-release algorithms with reproducible baselines, specifies actual exploration probabilities, and adds an implementation work plan and executable acceptance criteria.
