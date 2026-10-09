-- Migration 004: Create decisions table
-- SRS §9.1: Unique decision/request ID; canonical request hash, response, full candidate set/probabilities, versions, state watermark.

CREATE TABLE IF NOT EXISTS decisions (
    decision_id VARCHAR(128) PRIMARY KEY,
    request_id VARCHAR(128) NOT NULL UNIQUE,
    request_hash CHAR(64) NOT NULL,  -- SHA-256 hex digest
    seller_id VARCHAR(128) NOT NULL,
    lead_id VARCHAR(128) NOT NULL,
    response JSONB NOT NULL,
    candidate_timestamps JSONB NOT NULL,  -- Sorted list of eligible timestamps
    candidate_probabilities JSONB,  -- Corresponding probabilities
    assignment VARCHAR(16) NOT NULL,  -- CONTROL, TREATMENT, EXPLORE, SHADOW
    experiment_id VARCHAR(128),
    policy_version VARCHAR(64) NOT NULL,
    bundle_id VARCHAR(128) NOT NULL,
    model_compatibility_id VARCHAR(128) NOT NULL,
    profile_version VARCHAR(64),
    calendar_version VARCHAR(64),
    context_version INTEGER NOT NULL DEFAULT 1,
    state_version BIGINT NOT NULL,
    n_attempts INTEGER NOT NULL DEFAULT 0,
    n_eff INTEGER NOT NULL DEFAULT 0,
    prior_level FLOAT8,
    prior_weight FLOAT8,
    expected_reward FLOAT8,
    latent_std FLOAT8,
    predictive_std FLOAT8,
    candidate_count INTEGER NOT NULL DEFAULT 0,
    action_probability FLOAT8,
    assignment_probability FLOAT8,
    ope_eligible BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ NOT NULL
);

CREATE INDEX idx_decisions_seller_lead ON decisions(seller_id, lead_id);
CREATE INDEX idx_decisions_created ON decisions(created_at);
