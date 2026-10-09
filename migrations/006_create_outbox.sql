-- Migration 006: Create outbox table
-- SRS §9.1: Transactional events for committed state changes, decisions, and invalidations.

CREATE TABLE IF NOT EXISTS outbox (
    id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(32) NOT NULL,  -- OUTCOME_APPLIED, DECISION_CREATED, STATE_INVALIDATED
    event_id VARCHAR(128) NOT NULL,
    source VARCHAR(128) NOT NULL,
    attempt_id VARCHAR(128),
    seller_id VARCHAR(128) NOT NULL,
    model_compatibility_id VARCHAR(128) NOT NULL,
    state_version BIGINT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at TIMESTAMPTZ,
    retry_count INTEGER NOT NULL DEFAULT 0,
    UNIQUE (event_type, event_id)
);

CREATE INDEX idx_outbox_unprocessed ON outbox(event_type, created_at) WHERE processed_at IS NULL;
CREATE INDEX idx_outbox_seller ON outbox(seller_id, model_compatibility_id, state_version);

-- Migration tracking table
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

INSERT INTO schema_migrations (version) VALUES (1), (2), (3), (4), (5), (6)
ON CONFLICT (version) DO NOTHING;
