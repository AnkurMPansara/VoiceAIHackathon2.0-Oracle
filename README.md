# Best Time to Call & Dynamic Retry Engine

IndiaMART outbound calling intelligence system. Recommends optimal future call timestamps and dynamic retry schedules based on hierarchical Bayesian time-of-day models.

**SRS Version:** 2.0 &middot; **Target:** IndiaMART outbound calling infrastructure &middot; **Status:** Build phase &mdash; shadow testing ready

---

## Table of Contents

- [About The Project](#about-the-project)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Training the Model](#training-the-model)
  - [Phase 0: Data Quality Report](#phase-0-data-quality-report)
  - [Train Priors](#train-priors)
  - [Validate Bundle](#validate-bundle)
- [Running the Service](#running-the-service)
  - [Start API Server](#start-api-server)
  - [API Endpoints](#api-endpoints)
  - [Shadow Mode](#shadow-mode)
- [Single Seller Inference](#single-seller-inference)
- [CLI Interface](#cli-interface)
- [Evaluation](#evaluation)
  - [Predictive Backtest](#predictive-backtest)
  - [One-Step OPE](#one-step-ope)
  - [Benchmarks](#benchmarks)
- [Testing](#testing)
- [Data](#data)
- [Release Checklist](#release-checklist)
- [License](#license)

---

## About The Project

The Best Time to Call system learns a time-of-day reward curve from population segments and seller history, then recommends a concrete future call timestamp. After an outcome, it recommends the next permissible retry.

**Key features:**
- Hierarchical Bayesian model with Fourier basis for time-of-day patterns
- Population pooling for sparse seller history (average ~3.5 attempts per seller)
- Deterministic and exploration policies with explicit probabilities
- Dynamic retry engine respecting business calendar, caps, and seller preferences
- Idempotent outcome processing with revision handling
- Shadow mode for safe evaluation before live activation
- One-step counterfactual evaluation (IPS, SNIPS, DR)

**Does not:** place calls, reserve dialer capacity, use an LLM at runtime, or manage business policy.

---

## Architecture

```
best_time_to_call/
  pyproject.toml
  inference_for_seller.py     # Interactive single-seller inference script
  configs/
    development.yaml          # Dev defaults: shadow mode, 08-18h calendar
    production.example.yaml   # Template for production
  src/btc/
    config.py                 # Config loader, validation (CFG-01)
    schemas.py                # Pydantic data contracts (§4, §10)
    cli.py                    # CLI entry point (8 subcommands)
    data/
      adapters.py             # CSV readers, source field mapping (§4.4)
      normalization.py        # Timestamp localization, splits (TRAIN-06)
    features/
      fourier.py              # Fourier basis phi(t) (MOD-02)
    model/
      reward.py               # Reward computation (MOD-01)
      stats.py                # Seller sufficient statistics (MOD-04/05)
      posterior.py            # Posterior computation (MOD-06/07)
      priors.py               # Hierarchical prior fitting (TRAIN-01-05)
      policy.py               # Candidate generation, selection (POL-01-07)
      trainer.py              # Training pipeline, bundle management (TRAIN-07)
      bundle.py               # Bundle validation (TRAIN-07)
    store/
      database.py             # PostgreSQL/SQLite connection management
      transactions.py         # Atomic outcome processing (STATE-01-04)
      codec.py                # Compact binary cache codec (STATE-07)
      backfill.py             # Namespace backfill (STATE-08)
      reconciliation.py       # State reconciliation (STATE-05/08)
    retry/
      calendar.py             # Business calendar, support bins (POL-02, TRAIN-05)
      rules.py                # Dynamic retry decision table (RET-01-04)
    experiment/
      assignment.py           # SHA-256 experiment assignment (EXP-01)
      logging.py              # Durable decision logging (EXP-03)
    service/
      api.py                  # FastAPI endpoints (§10)
      benchmarks.py           # Performance benchmarks (NFR-05/06)
    evaluation/
      phase0.py               # Phase 0 data quality report (EVAL-01)
      replay.py               # Predictive backtest (EVAL-02)
      metrics.py              # OPE and online metrics (EVAL-03-06)
  migrations/                 # SQL migration files
  tests/
    unit/                     # 1047 unit tests
    integration/              # Integration tests
    failure_injection/        # Crash/recovery tests
    bench/                    # Performance benchmarks
  fixtures/synthetic/         # Test fixtures
  artifacts/example_bundle/   # Example model bundle
  docs/
    review_report.md          # Independent QA review
    decisions.md              # Design decisions
    runbook.md                # Operational procedures
    traceability.md           # Requirement-to-test mapping
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- PostgreSQL 14+ (production) or SQLite (development/testing)
- Optional: Redis 7+ (read cache)

```bash
# Install dependencies
pip install -r requirements.txt
# or
pip install numpy scipy pandas fastapi pydantic psycopg2-binary pyyaml pytest
```

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/best-time-to-call.git
cd best-time-to-call

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run database migrations
btc migrate-up

# Run tests
pytest tests/unit/ -v
```

### Configuration

Create `configs/development.yaml`:

```yaml
runtime_mode: shadow
timezone: Asia/Kolkata
calendar:
  days_of_week: [0, 1, 2, 3, 4, 5]  # Mon-Sat
  start_hour: 8
  end_hour: 18
  holidays: []
initial_delay_maximum_minutes: 15
max_calls_per_seller_per_day: 3
max_attempts_per_lead: 5
minimum_inter_call_gap_minutes: 15
dispatch_lead_time_seconds: 5
decision_execution_tolerance_minutes: 5
cache_staleness_seconds: 5
experiment_enabled: false
double_call_enabled: false
model_config:
  k: 4
  gamma: 1.0
  lambda_smooth: 1.0
  lambda_parent: 10.0
  alpha: 0.1
  sigma2: 1.0
  state_history_start: "2026-04-01"
  state_history_end: "2026-10-01"
reward_config:
  w_meeting: 1.0
  w_answered: 0.1
  c_dial: 0.02
  w_not_interested: 0.0
database:
  dsn: "postgresql://user:pass@localhost:5432/btc"
  pool_size: 10
```

---

## Training the Model

### Phase 0: Data Quality Report

Generate a descriptive statistics report on raw data before training:

```bash
btc phase0-report \
  --config configs/development.yaml \
  --attempts-csv data/Best-Time-to-Call\ -\ Call\ Attempts\ Apr-Sep\ 2026.csv \
  --sellers-csv data/Best-Time-to-Call\ -\ Sellers.csv \
  --output artifacts/phase0_report.json
```

This reports: distributions of observations per seller, hour-bin support, weekday, segments, call status, lead age, meeting/answer rates, duplicate frequency, and missing data.

### Train Priors

Train the hierarchical Bayesian model with hyperparameter grid search:

```bash
btc train-priors \
  --config configs/development.yaml \
  --data-dir data/ \
  --output artifacts/model_bundle \
  --dry-run
```

**What happens:**
1. Loads and normalizes CSV data
2. Creates chronological splits (Apr-Jul for priors, Jul-Aug for warmup, Aug-Sep for validation, Sep-Oct for test)
3. Computes segment statistics (cell &rarr; group_turnover &rarr; group &rarr; global)
4. Computes support bins (50 attempts, 30 sellers per 15-min bin)
5. Fits hierarchical priors with smoothness penalty
6. Grid searches K &in; {2,3,4}, lambda_smooth &in; {0.1,1,10}, lambda_parent &in; {10,100}, alpha &in; {0.01,0.1,1.0}
7. Selects best by Gaussian NLL on Aug-Sep validation
8. Creates immutable model bundle (JSON metadata + NPZ arrays)

### Validate Bundle

```bash
btc validate-bundle \
  --bundle-path artifacts/model_bundle \
  --verbose
```

Validates: dimensions, finite values, symmetry, positive-definiteness of precision matrices, checksums.

---

## Running the Service

### Start API Server

```bash
# Development (shadow mode)
uvicorn btc.service.api:app --host 0.0.0.0 --port 8000 --reload

# Production
uvicorn btc.service.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/v1/best-time` | POST | Get recommended call timestamp |
| `/v1/outcomes` | POST | Record finalized call outcome |
| `/v1/retries` | POST | Get retry recommendation |
| `/healthz` | GET | Process liveness |
| `/readyz` | GET | Readiness of model/config/stores |
| `/metrics` | GET | Operational metrics |

**Example: Get recommendation**

```bash
curl -X POST http://localhost:8000/v1/best-time \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-token-change-me" \
  -d '{
    "request_id": "req-001",
    "seller_id": "39323",
    "lead_id": "lead-001",
    "earliest_at": "2026-10-09T08:00:00+05:30",
    "latest_at": "2026-10-10T18:00:00+05:30"
  }'
```

**Example: Record outcome**

```bash
curl -X POST http://localhost:8000/v1/outcomes \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-token-change-me" \
  -d '{
    "seller_id": "39323",
    "lead_id": "lead-001",
    "attempt_id": "attempt-001",
    "source": "indiamart",
    "revision": 1,
    "event_id": "event-001",
    "finalized_at": "2026-10-09T12:00:00+05:30",
    "call_start_time": "2026-10-09T11:55:00+05:30",
    "call_end_time": "2026-10-09T11:56:00+05:30",
    "lead_sent_time": "2026-10-09T11:50:00+05:30",
    "attempt_number": 1,
    "answered": true,
    "disposition": "MEETING_FIXED",
    "meeting_fixed": true
  }'
```

**Example: Get retry recommendation**

```bash
curl -X POST http://localhost:8000/v1/retries \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dev-token-change-me" \
  -d '{
    "request_id": "retry-001",
    "source": "indiamart",
    "attempt_id": "attempt-001",
    "expected_revision": 1
  }'
```

### Shadow Mode

In shadow mode, recommendations are computed and logged but never submitted to the scheduler. This allows safe evaluation against real traffic without affecting calling behavior.

Set `runtime_mode: shadow` in config. To switch to live:

```yaml
runtime_mode: live
experiment_enabled: true
```

---

## Single Seller Inference

The `inference_for_seller.py` script provides an interactive way to run the Best Time to Call model for a single seller. It loads CSV data, builds seller state from historical attempts, computes the posterior, and displays candidate call slots with predicted rewards.

### Usage

```bash
python inference_for_seller.py
```

The script prompts for:
1. **Seller GLID** — the seller identifier to analyze
2. **Current time** — in `YYYY-MM-DD HH:MM` format (defaults to now if left blank)

### What it does

1. **Loads CSV data** — Reads call attempts and seller files from the `data/` directory
2. **Builds seller state** — Aggregates historical attempts for the seller into sufficient statistics
3. **Loads model bundle** — Loads trained priors and hyperparameters from `artifacts/model_bundle`
4. **Computes posterior** — Combines seller state with segment/global priors
5. **Generates candidate slots** — Creates 15-minute intervals over a 6-hour window from the current time
6. **Runs inference** — Calls `predict_for_time_slots` to compute expected rewards and uncertainties
7. **Displays results** — Shows candidate slots with IST timestamps, rewards, and historical attempts within the window

### Requirements

- A trained model bundle must exist at `artifacts/model_bundle`
- CSV data files must be present in the `data/` directory

If the model bundle is missing, train it first:

```bash
btc train-priors --config configs/development.yaml --data-dir data/ --output artifacts/model_bundle
```

### Example Output

```
======================================================================
BEST TIME TO CALL - Single Seller Inference
======================================================================
Please enter seller glid: 39323
Enter current time (YYYY-MM-DD HH:MM) [default: now]: 2026-10-10 10:00

Loading data...

======================================================================
INFERENCE RESULTS
======================================================================
Seller ID:          39323
Current Time:       2026-10-10 10:00:00 IST
Window:             6.0 hours
Cold Start:         False
Prior Weight:       0.8523
Historical Attempts:7
In Window:          3
Historical Meetings:2
Historical Rate:    0.2857

CANDIDATE SLOTS (15-min intervals):
----------------------------------------------------------------------
Time (IST)               Hour     Reward       Latent Std   Pred Std     Supported
----------------------------------------------------------------------
2026-10-10 10:00:00 IST  10.00    0.084321     0.123456     0.134567     Yes
2026-10-10 10:15:00 IST  10.25    0.091234     0.119876     0.131234     Yes
2026-10-10 10:30:00 IST  10.50    0.102345     0.115432     0.128765     Yes
...
----------------------------------------------------------------------

BEST SLOT: 2026-10-10 14:30:00 IST (Hour: 14.50, Reward: 0.125432)

HISTORICAL ATTEMPTS (within 6-hour window):
----------------------------------------------------------------------
Call Start (IST)           Hour     Answered   Meeting    Disposition
----------------------------------------------------------------------
2026-10-10 10:15:00 IST    10       Yes        Yes        MEETING_FIXED
2026-10-10 12:30:00 IST    12       Yes        No         INTERESTED
2026-10-10 14:00:00 IST    14       No         No         NOT_INTERESTED
----------------------------------------------------------------------
======================================================================
```

---

## CLI Interface

The `btc_cli.py` script provides a quick command-line interface for single-seller best time to call inference. It's designed for rapid testing and analysis with minimal setup.

### Usage

```bash
python btc_cli.py
```

Or with piped input for automation:

```bash
echo "39323
10:00
16:00" | python btc_cli.py
```

### Interactive Prompts

The script prompts for:
1. **Seller GLID** — the seller identifier to analyze
2. **Start time** — in `HH:MM` 24-hour format (e.g., `10:00`)
3. **End time** — in `HH:MM` 24-hour format (e.g., `16:00`)

Supports midnight-crossing windows (e.g., `22:00` to `04:00`).

### What it does

1. **Loads seller data** — Fetches seller's recent attempts from CSV (top 10)
2. **Displays recent attempts** — Shows last 10 calls with outcomes
3. **Analyzes time window** — Computes 15-minute interval slots between start and end times
4. **Runs inference** — Computes expected rewards for each slot
5. **Shows top 10 recommendations** — Sorted by expected reward (best to worst)
6. **Compares with history** — Shows historical best time in the window

### Output Format

```
================================================================================
BEST TIME TO CALL - CLI Interface
================================================================================

Please enter seller glid: 42689626
Loading data...
Found 10 recent attempts for seller 42689626

RECENT ATTEMPTS:
--------------------------------------------------------------------------------
Call Start (IST)          Answered   Meeting    Disposition         
--------------------------------------------------------------------------------
2026-06-02 16:06          Yes        Yes        MEETING_FIXED       
2026-06-11 15:52          Yes        Yes        MEETING_FIXED       
...

Enter start time (HH:MM, 24h format, e.g., 22:20): 14:00
Enter end time (HH:MM, 24h format, e.g., 6:00): 20:00
Window: 14:00 to 20:00 (6.0 hours)

================================================================================
TOP 10 RECOMMENDED SLOTS
================================================================================
Rank   Time (IST)                Hour     Reward       Latent Std   Pred Std    
--------------------------------------------------------------------------------
1      2026-10-10 20:00:00 IST   20:00    0.395714     0.392804     1.074381    
2      2026-10-10 19:45:00 IST   19:45    0.382785     0.386036     1.071925    
...
--------------------------------------------------------------------------------

RECOMMENDED BEST TIME: 2026-10-10 20:00:00 IST
  Hour: 20:00
  Expected Reward: 0.395714

HISTORICAL BEST IN WINDOW: 2026-06-02 16:06:31 IST
  Hour: 16:06
  Actual Reward: 1.080000

================================================================================
Seller: 42689626 | Cold Start: False | Prior Weight: 0.8081
Window: 14:00 to 20:00 | Slots: 25
================================================================================
```

### Requirements

- A trained model bundle must exist at `artifacts/model_bundle`
- CSV data files must be present in the `data/` directory

---

## Evaluation

### Predictive Backtest

Run chronological replay to evaluate model quality:

```bash
btc backtest \
  --config configs/development.yaml \
  --bundle-path artifacts/model_bundle \
  --data-dir data/ \
  --output artifacts/backtest_results.json
```

Reports: reward MSE, Gaussian predictive NLL, expected vs observed by time bin, results by history count groups (0, 1-4, 5-9, &ge;10).

### One-Step OPE

Evaluate policy lift using counterfactual estimation:

```bash
btc evaluate-ope \
  --config configs/development.yaml \
  --bundle-path artifacts/model_bundle \
  --decisions-json artifacts/logged_decisions.json \
  --output artifacts/ope_results.json
```

Reports: IPS, SNIPS, DR estimators, ESS, weight clipping, 95% CI from 1000 seller-cluster bootstrap replicates.

### Benchmarks

```bash
btc bench \
  --config configs/development.yaml \
  --workers 4 \
  --duration-seconds 900
```

Reports: p50/p95/p99 latency, throughput, errors, conflicts, CPU, RSS, cache hit rate.

---

## Testing

```bash
# All unit tests
pytest tests/unit/ -v

# Specific module
pytest tests/unit/test_fourier.py -v
pytest tests/unit/test_posterior.py -v
pytest tests/unit/test_policy.py -v

# Integration tests
pytest tests/integration/ -v

# Failure injection tests
pytest tests/failure_injection/ -v

# Benchmarks
pytest tests/bench/ -v
```

**Test coverage:** 1047 unit tests covering T01-T06, T08-T09, T14 from SRS &sect;15 acceptance matrix.

---

## Data

Raw data files are in the `data/` directory:

| File | Description |
|---|---|
| `Best-Time-to-Call - Call Attempts Apr-Sep 2026.csv` | 530,748 call attempts |
| `Best-Time-to-Call - Sellers.csv` | 149,363 seller profiles |
| `Best-Time-to-Call - Data Dictionary.md` | Field descriptions and mappings |

**Join key:** `fk_glusr_usr_id` (seller GLID)

**Data splits (TRAIN-06):**

| Purpose | Interval |
|---|---|
| Population prior fitting | 2026-04-01 to 2026-07-01 |
| Noise calibration & seller-state warmup | 2026-07-01 to 2026-08-01 |
| Hyperparameter validation | 2026-08-01 to 2026-09-01 |
| Final untouched test | 2026-09-01 to 2026-10-01 |

---

## Release Checklist

Before live activation, provide:

- [ ] Actual dial identity and revision semantics (data engineering)
- [ ] Versioned baseline adapter and dispatch contract (telephony engineering)
- [ ] Approved calling calendar and frequency limits (business policy)
- [ ] Fresh-lead delay and lead expiry semantics (product owner)
- [ ] Terminal disposition mappings (operations)
- [ ] Preregistered experiment metrics and guardrails (product/data science)
- [ ] Infrastructure workload profile and RPO/RTO (platform)

See SRS &sect;18 for full release inputs.

---

## License

Distributed under the MIT License. See `LICENSE.txt` for details.

---

*Built with Python, NumPy, SciPy, FastAPI, Pydantic, PostgreSQL. 1047 tests passing.*
