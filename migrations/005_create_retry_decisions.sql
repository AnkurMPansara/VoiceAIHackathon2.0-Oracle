-- Migration 005: Create retry_decisions table
-- SRS §9.1: Unique RET-04 key; result, supersession status, scheduler linkage.

CREATE TABLE IF NOT EXISTS retry_decisions (
    decision_id VARCHAR(128) PRIMARY KEY,
    source VARCHAR(128) NOT NULL,
    attempt_id VARCHAR(128) NOT NULL,
    revision INTEGER NOT NULL,
    retry_policy_version VARCHAR(64) NOT NULL,
    result VARCHAR(32) NOT NULL,  -- STOP, MANUAL_REVIEW, SUPERSEDED, SCHEDULED
    scheduled_at TIMESTAMPTZ,
    reason_code VARCHAR(64) NOT NULL,
    scheduler_decision_id VARCHAR(128),
    superseded_by VARCHAR(128),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source, attempt_id, revision, retry_policy_version)
);

CREATE INDEX idx_retry_decisions_attempt ON retry_decisions(source, attempt_id);
