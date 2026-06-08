from backend.server import _significant_signature


def test_significant_signature_ignores_volatile_fields():
    """Volatile per-tick noise (CPU/mem/timers/last_activity) must not flip the signature,
    so an otherwise-idle session stops emitting an SSE update every scan tick."""
    base = {
        "pid": 1,
        "status": "working",
        "usage": {"input_tokens": 5},
        "cpu_percent": 1.0,
        "memory_mb": 100.0,
        "duration_seconds": 10,
        "current_task_elapsed_seconds": 3,
        "last_activity_at": "2026-06-09T00:00:00Z",
    }
    a = _significant_signature(base)
    volatile_changed = _significant_signature(
        {
            **base,
            "cpu_percent": 99.0,
            "memory_mb": 999.0,
            "duration_seconds": 9999,
            "current_task_elapsed_seconds": 9999,
            "last_activity_at": "2026-06-09T01:00:00Z",
        }
    )
    assert a == volatile_changed  # noise alone → no broadcast


def test_significant_signature_detects_real_changes():
    base = {"pid": 1, "status": "working", "usage": {"input_tokens": 5}, "cpu_percent": 1.0}
    a = _significant_signature(base)
    assert _significant_signature({**base, "status": "idle"}) != a
    assert _significant_signature({**base, "usage": {"input_tokens": 6}}) != a
