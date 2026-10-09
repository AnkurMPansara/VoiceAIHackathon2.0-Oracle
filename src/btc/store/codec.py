"""Compact binary codec for SellerState serialization (STATE-07).

Implements the ≤1 KiB persistence format for seller sufficient statistics
as specified in SRS §9.2 (STATE-07, NFR-01, NFR-02).

Binary layout (little-endian):

    ┌─────────────────────────────────────────────────────────────────┐
    │ Header (48 bytes)                                               │
    ├─────────────────────────────────────────────────────────────────┤
    │ Payload: packed upper triangle of A (row-major) + b vector      │
    │         all float64, big-endian preserved via struct            │
    └─────────────────────────────────────────────────────────────────┘

Header fields (little-endian):

    Offset  Size  Field                  Type    Description
    ─────── ────  ───────────────────    ──────── ─────────────────
    0       4     magic                  u32     Must be 0x42544331 ("BTC1")
    4       2     format_version         u16     Must be 1
    6       2     d                      u16     Feature dimension
    8       8     compatibility_id       u64     Model compatibility tag
    16      8     n                      u64     Observation count
    24      8     state_version          u64     Monotonic state version
    32      8     last_commit_epoch_ms   i64     Millisecond timestamp
    40      4     crc32                  u32     CRC32 over header(CRC=0)+payload
    44      4     reserved               u32     Must be zero

Payload size formula:

    bytes(d) = 48 + 8 * (d*(d+1)/2 + d)

For K=4, d=9: bytes(9) = 48 + 8 * (45 + 9) = 480 bytes.

Modules
-------
encode_state      : SellerState → bytes
decode_state      : bytes → (SellerState, compatibility_id)
compute_crc32     : bytes → CRC32 checksum
compute_payload_size : d → payload byte length
validate_payload  : bytes → validation report (no full decode)
state_to_dict     : SellerState → JSON-serializable dict
dict_to_state     : dict → SellerState
"""

from __future__ import annotations

import struct
import zlib
from typing import Any

import numpy as np

from btc.model.stats import SellerState

# ── Constants ─────────────────────────────────────────────────────────────────

MAGIC: int = 0x42544331  # "BTC1"
FORMAT_VERSION: int = 1
HEADER_SIZE: int = 48

# Struct format: all little-endian, no padding
# magic(u32) + format_version(u16) + d(u16) + compatibility_id(u64) +
# n(u64) + state_version(u64) + last_commit_epoch_ms(i64) + crc32(u32) + reserved(u32)
_HEADER_FORMAT: str = "<IHHQQQqII"

# Signed timestamp bounds (epoch milliseconds)
_MIN_TIMESTAMP_MS: int = -9223372036854775808  # i64 min
_MAX_TIMESTAMP_MS: int = 253402300799999  # year 9999-12-31, practical upper bound

# Unsigned counter bounds
_MAX_COUNTER: int = 18446744073709551615  # u64 max


# ── CRC32 ─────────────────────────────────────────────────────────────────────


def compute_crc32(data: bytes) -> int:
    """Compute CRC32 checksum over *data*.

    Parameters
    ----------
    data : bytes
        The data to checksum.

    Returns
    -------
    int
        Unsigned 32-bit CRC32 value (0 ≤ value ≤ 0xFFFFFFFF).
    """
    return zlib.crc32(data) & 0xFFFFFFFF


# ── Payload size ──────────────────────────────────────────────────────────────


def compute_payload_size(d: int) -> int:
    """Compute serialized payload size (in bytes) for dimension *d*.

    Formula: bytes(d) = 48 + 8 * (d*(d+1)/2 + d)

    Parameters
    ----------
    d : int
        Feature dimension (must be positive).

    Returns
    -------
    int
        Total serialized size including 48-byte header.

    Raises
    ------
    ValueError
        If *d* is not a positive integer.
    """
    if not isinstance(d, int) or d <= 0:
        raise ValueError(f"d must be a positive integer, got {d!r}")

    upper_elements: int = d * (d + 1) // 2
    b_elements: int = d
    return HEADER_SIZE + 8 * (upper_elements + b_elements)


# ── Encoding ──────────────────────────────────────────────────────────────────


def encode_state(
    state: SellerState,
    compatibility_id: int,
    last_commit_epoch_ms: int,
) -> bytes:
    """Encode a :class:`SellerState` to its compact binary representation.

    STATE-07: Produces a bytes object of exactly ``compute_payload_size(state.d)``
    length, containing a 48-byte header followed by packed float64 values for
    the upper triangle of A and the b vector.

    Parameters
    ----------
    state : SellerState
        The seller state to serialize. Must have ``d > 0``, finite float64
        arrays, and valid ``n`` / ``state_version`` counters.
    compatibility_id : int
        Model compatibility tag. Stored as unsigned 64-bit integer.
        Must be in ``[0, 2^64 - 1]``.
    last_commit_epoch_ms : int
        Millisecond epoch of the last commit. Stored as signed 64-bit
        integer. Must be in ``[i64_min, i64_max]``.

    Returns
    -------
    bytes
        Packed binary payload of length ``compute_payload_size(state.d)``.

    Raises
    ------
    ValueError
        If any field is nonfinite, dimensions mismatch, counters are out of
        range, or the payload would exceed 1 KiB.
    TypeError
        If *state* is not a :class:`SellerState`.
    """
    if not isinstance(state, SellerState):
        raise TypeError(f"Expected SellerState, got {type(state).__name__}")

    d: int = state.d
    expected_total: int = compute_payload_size(d)

    if expected_total > 1024:
        raise ValueError(
            f"Serialized size {expected_total} exceeds 1 KiB limit for d={d}"
        )

    # Validate array shapes
    expected_upper_size: int = d * (d + 1) // 2
    if state.A_upper.shape != (expected_upper_size,):
        raise ValueError(
            f"A_upper shape must be ({expected_upper_size},), "
            f"got {state.A_upper.shape}"
        )
    if state.b.shape != (d,):
        raise ValueError(f"b shape must be ({d},), got {state.b.shape}")

    # Validate float64 dtype
    if state.A_upper.dtype != np.float64:
        raise ValueError("A_upper must be dtype float64")
    if state.b.dtype != np.float64:
        raise ValueError("b must be dtype float64")

    # Validate finite values
    if not np.all(np.isfinite(state.A_upper)):
        raise ValueError("A_upper contains nonfinite values")
    if not np.all(np.isfinite(state.b)):
        raise ValueError("b contains nonfinite values")

    # Validate counters
    if not (0 <= state.n <= _MAX_COUNTER):
        raise ValueError(f"n={state.n} out of unsigned 64-bit range")
    if not (0 <= state.state_version <= _MAX_COUNTER):
        raise ValueError(
            f"state_version={state.state_version} out of unsigned 64-bit range"
        )

    # Validate compatibility_id
    if not (0 <= compatibility_id <= _MAX_COUNTER):
        raise ValueError(
            f"compatibility_id={compatibility_id} out of unsigned 64-bit range"
        )

    # Validate timestamp
    if not (_MIN_TIMESTAMP_MS <= last_commit_epoch_ms <= _MAX_TIMESTAMP_MS):
        raise ValueError(
            f"last_commit_epoch_ms={last_commit_epoch_ms} out of signed 64-bit range"
        )

    # Pack header with CRC placeholder = 0
    header = struct.pack(
        _HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        d,
        compatibility_id,
        state.n,
        state.state_version,
        last_commit_epoch_ms,
        0,  # CRC placeholder
        0,  # reserved
    )

    # Pack payload: upper triangle of A, then b
    payload: bytes = state.A_upper.tobytes(order="C") + state.b.tobytes(order="C")

    # Compute CRC32 over header (with CRC zeroed) + payload
    crc: int = compute_crc32(header + payload)

    # Repack header with actual CRC
    header = struct.pack(
        _HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        d,
        compatibility_id,
        state.n,
        state.state_version,
        last_commit_epoch_ms,
        crc,
        0,  # reserved
    )

    return header + payload


# ── Decoding ──────────────────────────────────────────────────────────────────


def decode_state(payload: bytes) -> tuple[SellerState, int]:
    """Decode a binary payload to a :class:`SellerState`.

    STATE-07: Validates the magic number, format version, total length,
    CRC32 checksum, dimension consistency, reserved bits, timestamp range,
    counter range, and finite float64 values.

    Parameters
    ----------
    payload : bytes
        Binary data produced by :func:`encode_state`. Must be at least
        ``HEADER_SIZE`` bytes.

    Returns
    -------
    tuple[SellerState, int]
        ``(state, compatibility_id)`` where *state* is the reconstructed
        :class:`SellerState` and *compatibility_id* is the tag from the header.

    Raises
    ------
    ValueError
        If any validation check fails:
        - Wrong magic number
        - Unsupported format version
        - Incorrect total length for the stated dimension
        - CRC32 mismatch
        - Non-zero reserved bits
        - Dimension out of range
        - Timestamp or counter out of range
        - Nonfinite payload values
        - Array shape mismatch
    TypeError
        If *payload* is not bytes-like.
    """
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(payload).__name__}")

    # Check minimum length
    if len(payload) < HEADER_SIZE:
        raise ValueError(
            f"Payload too short: {len(payload)} bytes, minimum {HEADER_SIZE}"
        )

    # Unpack header
    fields: tuple = struct.unpack(
        _HEADER_FORMAT, payload[:HEADER_SIZE]
    )

    (
        magic,
        format_version,
        d,
        compatibility_id,
        n,
        state_version,
        last_commit_epoch_ms,
        crc_stored,
        reserved,
    ) = fields

    # ── Validate magic ──────────────────────────────────────────────────
    if magic != MAGIC:
        raise ValueError(
            f"Invalid magic: 0x{magic:08X}, expected 0x{MAGIC:08X} (BTC1)"
        )

    # ── Validate format version ─────────────────────────────────────────
    if format_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version: {format_version}, expected {FORMAT_VERSION}"
        )

    # ── Validate dimension ──────────────────────────────────────────────
    if not isinstance(d, int) or d <= 0:
        raise ValueError(f"Invalid dimension: d={d!r}")

    expected_total: int = compute_payload_size(d)
    if len(payload) != expected_total:
        raise ValueError(
            f"Payload length {len(payload)} does not match "
            f"expected {expected_total} for d={d}"
        )

    # ── Validate reserved bits ──────────────────────────────────────────
    if reserved != 0:
        raise ValueError(f"Reserved bits must be zero, got {reserved}")

    # ── Validate timestamp range ────────────────────────────────────────
    if not (_MIN_TIMESTAMP_MS <= last_commit_epoch_ms <= _MAX_TIMESTAMP_MS):
        raise ValueError(
            f"last_commit_epoch_ms={last_commit_epoch_ms} "
            f"out of valid range [{_MIN_TIMESTAMP_MS}, {_MAX_TIMESTAMP_MS}]"
        )

    # ── Validate unsigned counters ──────────────────────────────────────
    if not (0 <= n <= _MAX_COUNTER):
        raise ValueError(f"n={n} out of unsigned 64-bit range")
    if not (0 <= state_version <= _MAX_COUNTER):
        raise ValueError(
            f"state_version={state_version} out of unsigned 64-bit range"
        )

    # ── Validate CRC32 ──────────────────────────────────────────────────
    # Zero out the CRC field in the header for verification
    header_for_crc = bytearray(payload[:HEADER_SIZE])
    # CRC is at offset 40, 4 bytes little-endian
    struct.pack_into("<I", header_for_crc, 40, 0)

    crc_computed: int = compute_crc32(bytes(header_for_crc) + payload[HEADER_SIZE:])
    if crc_stored != crc_computed:
        raise ValueError(
            f"CRC32 mismatch: stored=0x{crc_stored:08X}, "
            f"computed=0x{crc_computed:08X}"
        )

    # ── Unpack payload ──────────────────────────────────────────────────
    payload_data: bytes = payload[HEADER_SIZE:]
    upper_size: int = d * (d + 1) // 2
    payload_bytes: int = 8 * (upper_size + d)

    if len(payload_data) != payload_bytes:
        raise ValueError(
            f"Payload data length {len(payload_data)} does not match "
            f"expected {payload_bytes} for d={d}"
        )

    # Unpack upper triangle of A
    A_upper = np.frombuffer(payload_data[: upper_size * 8], dtype=np.float64)

    # Unpack b vector
    b = np.frombuffer(payload_data[upper_size * 8 :], dtype=np.float64)

    # ── Validate finite values ──────────────────────────────────────────
    if not np.all(np.isfinite(A_upper)):
        raise ValueError("A_upper contains nonfinite values (inf or NaN)")
    if not np.all(np.isfinite(b)):
        raise ValueError("b contains nonfinite values (inf or NaN)")

    # ── Validate array shapes ───────────────────────────────────────────
    if A_upper.shape != (upper_size,):
        raise ValueError(
            f"A_upper shape ({A_upper.shape}) does not match "
            f"expected ({upper_size},)"
        )
    if b.shape != (d,):
        raise ValueError(f"b shape ({b.shape}) does not match expected ({d},)")

    # ── Reconstruct SellerState ─────────────────────────────────────────
    state: SellerState = SellerState(
        A_upper=A_upper,
        b=b,
        n=int(n),
        state_version=int(state_version),
        d=int(d),
    )

    return (state, int(compatibility_id))


# ── Validation (no full decode) ───────────────────────────────────────────────


def validate_payload(payload: bytes) -> dict:
    """Validate a binary payload without fully decoding it.

    Returns a report describing what was checked and the results.
    This is useful for pre-flight checks on cached data.

    Parameters
    ----------
    payload : bytes
        Binary data to validate.

    Returns
    -------
    dict
        Validation report with keys:

        - ``valid`` (bool): Whether all checks passed.
        - ``header`` (dict | None): Unpacked header fields, or ``None`` if
          the header could not be parsed.
        - ``errors`` (list[str]): List of validation error messages.
        - ``payload_size`` (int | None): Total payload length.
        - ``dimension`` (int | None): Extracted dimension *d*, or ``None``.
        - ``expected_total`` (int | None): Expected total size for *d*,
          or ``None``.
        - ``crc_match`` (bool | None): CRC32 match result, or ``None``.
    """
    report: dict[str, Any] = {
        "valid": False,
        "header": None,
        "errors": [],
        "payload_size": len(payload) if isinstance(payload, (bytes, bytearray)) else None,
        "dimension": None,
        "expected_total": None,
        "crc_match": None,
    }

    if not isinstance(payload, (bytes, bytearray)):
        report["errors"].append("Payload is not bytes-like")
        return report

    # Check minimum length
    if len(payload) < HEADER_SIZE:
        report["errors"].append(
            f"Payload too short: {len(payload)} bytes, minimum {HEADER_SIZE}"
        )
        return report

    # Unpack header
    try:
        fields = struct.unpack(_HEADER_FORMAT, payload[:HEADER_SIZE])
    except struct.error as exc:
        report["errors"].append(f"Failed to unpack header: {exc}")
        return report

    (
        magic,
        format_version,
        d,
        compatibility_id,
        n,
        state_version,
        last_commit_epoch_ms,
        crc_stored,
        reserved,
    ) = fields

    header_report: dict[str, Any] = {
        "magic": f"0x{magic:08X}",
        "format_version": format_version,
        "d": d,
        "compatibility_id": compatibility_id,
        "n": n,
        "state_version": state_version,
        "last_commit_epoch_ms": last_commit_epoch_ms,
        "crc32": f"0x{crc_stored:08X}",
        "reserved": reserved,
    }

    # Validate magic
    if magic != MAGIC:
        report["errors"].append(
            f"Invalid magic: 0x{magic:08X}, expected 0x{MAGIC:08X}"
        )

    # Validate format version
    if format_version != FORMAT_VERSION:
        report["errors"].append(
            f"Unsupported format version: {format_version}"
        )

    # Validate dimension
    if not isinstance(d, int) or d <= 0:
        report["errors"].append(f"Invalid dimension: d={d!r}")
    else:
        header_report["d"] = d
        report["dimension"] = d
        expected_total = compute_payload_size(d)
        header_report["expected_total"] = expected_total
        report["expected_total"] = expected_total

        if len(payload) != expected_total:
            report["errors"].append(
                f"Length {len(payload)} != expected {expected_total} for d={d}"
            )

    # Validate reserved bits
    if reserved != 0:
        report["errors"].append(f"Reserved bits non-zero: {reserved}")

    # Validate timestamp
    if not (_MIN_TIMESTAMP_MS <= last_commit_epoch_ms <= _MAX_TIMESTAMP_MS):
        report["errors"].append(
            f"Timestamp out of range: {last_commit_epoch_ms}"
        )

    # Validate counters
    if not (0 <= n <= _MAX_COUNTER):
        report["errors"].append(f"n out of range: {n}")
    if not (0 <= state_version <= _MAX_COUNTER):
        report["errors"].append(f"state_version out of range: {state_version}")

    # Validate CRC32
    if len(payload) >= HEADER_SIZE:
        header_for_crc = bytearray(payload[:HEADER_SIZE])
        struct.pack_into("<I", header_for_crc, 40, 0)
        crc_computed = compute_crc32(bytes(header_for_crc) + payload[HEADER_SIZE:])
        crc_match = crc_stored == crc_computed
        report["crc_match"] = crc_match
        header_report["crc32_computed"] = f"0x{crc_computed:08X}"

        if not crc_match:
            report["errors"].append(
                f"CRC32 mismatch: stored=0x{crc_stored:08X}, "
                f"computed=0x{crc_computed:08X}"
            )

    report["header"] = header_report
    report["valid"] = len(report["errors"]) == 0

    return report


# ── Dict conversion ───────────────────────────────────────────────────────────


def state_to_dict(state: SellerState) -> dict[str, Any]:
    """Convert a :class:`SellerState` to a JSON-serializable dictionary.

    All numpy arrays are converted to lists of Python floats. The *d* field
    is included for round-trip reconstruction.

    Parameters
    ----------
    state : SellerState
        The state to convert.

    Returns
    -------
    dict
        Keys: ``A_upper``, ``b``, ``n``, ``state_version``, ``d``.
        Array values are ``list[float]``.
    """
    return {
        "A_upper": state.A_upper.tolist(),
        "b": state.b.tolist(),
        "n": int(state.n),
        "state_version": int(state.state_version),
        "d": int(state.d),
    }


def dict_to_state(d: int, data: dict[str, Any]) -> SellerState:
    """Convert a dictionary back to a :class:`SellerState`.

    Parameters
    ----------
    d : int
        Feature dimension. Must match ``data["d"]`` if present.
    data : dict
        Dictionary with keys ``A_upper``, ``b``, ``n``, ``state_version``,
        and optionally ``d``. Arrays may be ``list[float]`` or ``np.ndarray``.

    Returns
    -------
    SellerState
        Reconstructed state with validated dimensions.

    Raises
    ------
    ValueError
        If required keys are missing, dimensions mismatch, or values
        are not finite.
    """
    required_keys = ("A_upper", "b", "n", "state_version")
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing required key: {key!r}")

    # Dimension consistency
    if "d" in data and data["d"] != d:
        raise ValueError(
            f"Dimension mismatch: data['d']={data['d']}, expected d={d}"
        )

    # Extract arrays
    A_upper_raw = data["A_upper"]
    b_raw = data["b"]

    A_upper = np.asarray(A_upper_raw, dtype=np.float64)
    b = np.asarray(b_raw, dtype=np.float64)

    # Validate shapes
    expected_upper_size = d * (d + 1) // 2
    if A_upper.shape != (expected_upper_size,):
        raise ValueError(
            f"A_upper shape ({A_upper.shape}) does not match "
            f"expected ({expected_upper_size},) for d={d}"
        )
    if b.shape != (d,):
        raise ValueError(f"b shape ({b.shape}) does not match expected ({d},)")

    # Validate finite
    if not np.all(np.isfinite(A_upper)):
        raise ValueError("A_upper contains nonfinite values")
    if not np.all(np.isfinite(b)):
        raise ValueError("b contains nonfinite values")

    return SellerState(
        A_upper=A_upper,
        b=b,
        n=int(data["n"]),
        state_version=int(data["state_version"]),
        d=d,
    )
