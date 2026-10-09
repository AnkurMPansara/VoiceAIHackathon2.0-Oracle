-- Migration 002: Create attempt_revisions audit table
-- SRS §9.1: Append-only audit of accepted revisions with ingestion time and payload.

CREATE TABLE IF NOT EXISTS attempt_revisions (
    id BIGSERIAL PRIMARY KEY,
    source VARCHAR(128) NOT NULL,
    attempt_id VARCHAR(128) NOT NULL,
    revision INTEGER NOT NULL,
    seller_id VARCHAR(128) NOT NULL,
    lead_id VARCHAR(128) NOT NULL,
    payload_hash CHAR(64) NOT NULL,
    phi_features JSONB,  -- Fourier feature vector
    reward_value FLOAT8,
    finalized_at TIMESTAMPTZ NOT NULL,
    call_start_time TIMESTAMPTZ NOT NULL,
    call_end_time TIMESTAMPTZ NOT NULL,
    lead_sent_time TIMESTAMPTZ NOT NULL,
    attempt_number INTEGER NOT NULL,
    answered BOOLEAN NOT NULL,
    disposition VARCHAR(32) NOT NULL,
    meeting_fixed BOOLEAN NOT NULL,
    commit_time TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ingestion_time TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_revisions_source_attempt ON attempt_revisions(source, attempt_id);
CREATE INDEX idx_revisions_seller ON attempt_revisions(seller_id);
CREATE INDEX idx_revisions_commit ON attempt_revisions(commit_time);
