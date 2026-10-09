-- Migration 003: Create seller_model_state table
-- SRS §9.1: PK (model_compatibility_id, seller_id); A, b, n, state_version, max call time, last commit time.

CREATE TABLE IF NOT EXISTS seller_model_state (
    model_compatibility_id VARCHAR(128) NOT NULL,
    seller_id VARCHAR(128) NOT NULL,
    state_version BIGINT NOT NULL DEFAULT 0,
    n_attempts INTEGER NOT NULL DEFAULT 0,
    a_upper BYTEA NOT NULL,  -- Packed upper triangle of A matrix
    b_vector BYTEA NOT NULL,  -- Packed b vector
    max_call_time TIMESTAMPTZ,
    last_commit_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (model_compatibility_id, seller_id)
);

CREATE INDEX idx_seller_state_seller ON seller_model_state(seller_id);
