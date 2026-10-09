"""Retry scheduling package for the Best Time to Call system.

Submodules
----------
calendar
    Business calendar and supported-hours computation (POL-02, POL-03, TRAIN-05).
rules
    Dynamic retry decision engine (RET-01 through RET-04).
"""

from btc.retry.calendar import (
    BusinessCalendar,
    compute_support_mask,
    find_next_available_slot,
    is_within_call_window,
    is_working_day,
    next_working_day,
)
from btc.retry.rules import (
    RetryPlan,
    is_terminal_disposition,
    plan_retry,
    plan_retry_idempotent,
    project_to_eligible,
    resolve_callback,
    search_curve_peaks,
)

__all__ = [
    # calendar
    "BusinessCalendar",
    "compute_support_mask",
    "find_next_available_slot",
    "is_within_call_window",
    "is_working_day",
    "next_working_day",
    # rules
    "RetryPlan",
    "is_terminal_disposition",
    "plan_retry",
    "plan_retry_idempotent",
    "project_to_eligible",
    "resolve_callback",
    "search_curve_peaks",
]
