-- Migration 001: Create attempt_latest table
-- SRS §9.1: PK (source, attempt_id); seller/lead identity, revision, canonical payload hash, finalized labels and times.

CREATE TABLE IF NOT EXISTS attempt_latest (
    source VARCHAR(128) NOT NULL,
    attempt_id VARCHAR(128) NOT NULL,
    seller_id VARCHAR(128) NOT NULL,
    lead_id VARCHAR(128) NOT NULL,
    revision INTEGER NOT NULL DEFAULT 1,
    payload_hash CHAR(64) NOT NULL,  -- SHA-256 hex digest
    finalized_at TIMESTAMPTZ NOT NULL,
    call_start_time TIMESTAMPTZ NOT NULL,
    call_end_time TIMESTAMPTZ NOT NULL,
    lead_sent_time TIMESTAMPTZ NOT NULL,
    attempt_number INTEGER NOT NULL,
    answered BOOLEAN NOT NULL,
    disposition VARCHAR(32) NOT NULL,
    meeting_fixed BOOLEAN NOT NULL,
    requested_callback_at TIMESTAMPTZ,
    decision_id VARCHAR(128),
    duration_s INTEGER,
    dialer_version VARCHAR(64),
    source_bucket VARCHAR(64),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (source, attempt_id)
);

CREATE INDEX idx_attempt_latest_seller ON attempt_latest(seller_id);
CREATE INDEX idx_attempt_latest_lead ON attempt_latest(lead_id);
CREATE INDEX idx_attempt_latest_finalized ON attempt_latest(finalized_at);
