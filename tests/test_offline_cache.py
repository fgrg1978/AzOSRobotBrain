# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Tests for offline_cache.ModelCache (B03)."""

import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from offline_cache import (  # noqa: E402
    ModelCache,
    CacheEntry,
    CacheStats,
    FALLBACK_RESPONSES,
    FALLBACK_DEFAULT_CLASSIFICATION,
    OFFLINE_CACHE_MAX_ENTRIES,
    OFFLINE_CACHE_TTL_S,
    OFFLINE_CACHE_SCHEMA_VERSION,
    OFFLINE_CACHE_KEY_HEX_LEN,
)

# ── Constants / smoke ────────────────────────────────────────────────────────


def test_constants_are_positive():
    assert OFFLINE_CACHE_MAX_ENTRIES > 0
    assert OFFLINE_CACHE_TTL_S > 0
    assert OFFLINE_CACHE_SCHEMA_VERSION >= 1
    assert OFFLINE_CACHE_KEY_HEX_LEN > 0


def test_fallback_catalog_has_defaults():
    for kind in ("llm", "vlm", "task"):
        assert kind in FALLBACK_RESPONSES
        assert FALLBACK_DEFAULT_CLASSIFICATION in FALLBACK_RESPONSES[kind]


# ── Keys ─────────────────────────────────────────────────────────────────────


def test_make_key_deterministic():
    k1 = ModelCache.make_key("vlm", "hello")
    k2 = ModelCache.make_key("vlm", "hello")
    assert k1 == k2
    assert len(k1) == OFFLINE_CACHE_KEY_HEX_LEN


def test_make_key_differs_on_kind():
    assert ModelCache.make_key("vlm", "x") != ModelCache.make_key("llm", "x")


def test_make_key_differs_on_prompt():
    assert ModelCache.make_key("vlm", "a") != ModelCache.make_key("vlm", "b")


def test_make_key_with_image_bytes():
    k_noimg = ModelCache.make_key("vlm", "p")
    k_img = ModelCache.make_key("vlm", "p", image_bytes=b"\x00\x01\x02")
    assert k_noimg != k_img


def test_make_key_with_extra():
    k1 = ModelCache.make_key("vlm", "p", extra="ctx-a")
    k2 = ModelCache.make_key("vlm", "p", extra="ctx-b")
    assert k1 != k2


def test_hash_bytes_stable():
    a = ModelCache.hash_bytes(b"hello")
    b = ModelCache.hash_bytes(b"hello")
    assert a == b
    assert a != ModelCache.hash_bytes(b"world")


# ── Put / Get ────────────────────────────────────────────────────────────────


@pytest.fixture
def cache(tmp_path):
    return ModelCache(
        path=str(tmp_path / "cache.json"),
        max_entries=4,
        ttl_s=60.0,
    )


def test_put_and_get(cache):
    key = ModelCache.make_key("llm", "prompt-1")
    assert cache.get(key) is None
    cache.put(key, "FORWARD 50", kind="llm")
    assert cache.get(key) == "FORWARD 50"


def test_get_miss_increments_stats(cache):
    cache.get("does-not-exist")
    s = cache.stats()
    assert s.misses == 1
    assert s.hits == 0


def test_get_hit_increments_stats(cache):
    k = cache.remember("llm", "p", "STOP")
    cache.get(k)
    cache.get(k)
    s = cache.stats()
    assert s.hits == 2
    assert s.misses == 0


def test_remember_and_lookup(cache):
    cache.remember("vlm", "describe this", "a wall")
    assert cache.lookup("vlm", "describe this") == "a wall"
    assert cache.lookup("vlm", "something else") is None


def test_prompt_deduplication(cache):
    # Two identical prompts -> one stored entry, first call cached.
    cache.remember("llm", "same-prompt", "ACTION_A")
    cache.remember("llm", "same-prompt", "ACTION_A")  # overwrite
    assert cache.size == 1
    assert cache.lookup("llm", "same-prompt") == "ACTION_A"


def test_put_refreshes_existing(cache):
    k = cache.remember("llm", "p", "v1")
    cache.put(k, "v2", kind="llm")
    assert cache.get(k) == "v2"
    assert cache.size == 1


# ── TTL / expiration ─────────────────────────────────────────────────────────


def test_ttl_expiration(tmp_path):
    c = ModelCache(path=str(tmp_path / "c.json"), max_entries=4, ttl_s=0.05)
    k = c.remember("llm", "p", "X")
    time.sleep(0.08)
    assert c.get(k) is None
    s = c.stats()
    assert s.expirations >= 1


def test_purge_expired(tmp_path):
    c = ModelCache(path=str(tmp_path / "c.json"), max_entries=16, ttl_s=0.05)
    c.remember("llm", "p1", "A")
    c.remember("llm", "p2", "B")
    time.sleep(0.08)
    removed = c.purge_expired()
    assert removed == 2
    assert c.size == 0


# ── LRU eviction ─────────────────────────────────────────────────────────────


def test_lru_eviction(tmp_path):
    c = ModelCache(path=str(tmp_path / "c.json"), max_entries=3, ttl_s=60.0)
    k1 = c.remember("llm", "p1", "r1")
    k2 = c.remember("llm", "p2", "r2")
    k3 = c.remember("llm", "p3", "r3")
    # Touch k1 so it's MRU.
    c.get(k1)
    # Insert p4 — k2 should be evicted (least recently used).
    c.remember("llm", "p4", "r4")
    assert c.get(k1) == "r1"
    assert c.get(k2) is None
    assert c.get(k3) == "r3"
    assert c.stats().evictions == 1


def test_eviction_count_tracks_overflow(tmp_path):
    c = ModelCache(path=str(tmp_path / "c.json"), max_entries=2, ttl_s=60.0)
    for i in range(5):
        c.remember("llm", f"p{i}", f"r{i}")
    assert c.size == 2
    assert c.stats().evictions == 3


# ── Stats ────────────────────────────────────────────────────────────────────


def test_hit_rate_zero_when_empty(cache):
    assert cache.stats().hit_rate() == 0.0


def test_hit_rate_basic(cache):
    k = cache.remember("llm", "p", "X")
    cache.get(k)  # hit
    cache.get("missing")  # miss
    assert abs(cache.stats().hit_rate() - 0.5) < 1e-6


def test_stats_as_dict(cache):
    d = cache.stats().as_dict()
    for key in ("hits", "misses", "evictions", "expirations", "fallbacks", "writes", "hit_rate"):
        assert key in d


def test_reset_stats(cache):
    k = cache.remember("llm", "p", "X")
    cache.get(k)
    cache.reset_stats()
    s = cache.stats()
    assert s.hits == 0 and s.misses == 0 and s.writes == 0


# ── Fallback responses ───────────────────────────────────────────────────────


def test_fallback_known_classification(cache):
    resp = cache.fallback("llm", classification="obstacle")
    assert resp == FALLBACK_RESPONSES["llm"]["obstacle"]


def test_fallback_unknown_classification_falls_to_default(cache):
    resp = cache.fallback("llm", classification="not-a-real-class")
    assert resp == FALLBACK_RESPONSES["llm"][FALLBACK_DEFAULT_CLASSIFICATION]


def test_fallback_unknown_kind_returns_safe_stop(cache):
    resp = cache.fallback("nonexistent-kind")
    assert resp == "STOP"


def test_fallback_increments_stats(cache):
    cache.fallback("llm")
    cache.fallback("vlm", classification="unknown")
    assert cache.stats().fallbacks == 2


def test_fallback_vlm_unknown_is_conservative(cache):
    # "unknown" classification should recommend caution.
    resp = cache.fallback("vlm", classification="unknown")
    assert "caution" in resp.lower() or "obstacle" in resp.lower()


# ── Decorator ────────────────────────────────────────────────────────────────


def test_cached_decorator_returns_cached_value(cache):
    calls = {"n": 0}

    @cache.cached(kind="llm")
    def fake_llm(prompt: str) -> str:
        calls["n"] += 1
        return f"response:{prompt}"

    r1 = fake_llm("hello")
    r2 = fake_llm("hello")
    assert r1 == r2 == "response:hello"
    assert calls["n"] == 1  # second call was cached


def test_cached_decorator_different_args_new_call(cache):
    calls = {"n": 0}

    @cache.cached(kind="llm")
    def fake(x: str) -> str:
        calls["n"] += 1
        return f"r:{x}"

    fake("a")
    fake("b")
    assert calls["n"] == 2


def test_cached_decorator_custom_key(cache):
    calls = {"n": 0}

    @cache.cached(kind="llm", key_from_args=lambda *a, **k: a[0])
    def fake(prompt: str, tag: str) -> str:
        calls["n"] += 1
        return f"x:{prompt}"

    # Same prompt, different tag -> still cached because key_from_args only uses prompt.
    fake("p", "t1")
    fake("p", "t2")
    assert calls["n"] == 1


def test_cached_decorator_serialises_non_str(cache):
    @cache.cached(kind="task")
    def fake() -> dict:
        return {"skill": "STOP"}

    r1 = fake()
    r2 = fake()
    # Second call returns the JSON-serialised cached string.
    assert r1 == {"skill": "STOP"}
    assert r2 == json.dumps({"skill": "STOP"})


# ── Persistence ──────────────────────────────────────────────────────────────


def test_save_and_reload(tmp_path):
    path = str(tmp_path / "cache.json")
    c1 = ModelCache(path=path, max_entries=8, ttl_s=3600.0)
    c1.remember("llm", "prompt-A", "FORWARD 50")
    c1.remember("vlm", "prompt-B", "clear path")
    c1.set_model_manifest({"llm": "llama-3.2-3b", "vlm": "smolvlm"})
    c1.save()

    assert os.path.exists(path)

    c2 = ModelCache.load(path=path, max_entries=8, ttl_s=3600.0)
    assert c2.size == 2
    assert c2.lookup("llm", "prompt-A") == "FORWARD 50"
    assert c2.lookup("vlm", "prompt-B") == "clear path"
    assert c2.get_model_manifest()["vlm"] == "smolvlm"


def test_load_missing_file_is_empty(tmp_path):
    c = ModelCache.load(path=str(tmp_path / "nothere.json"))
    assert c.size == 0


def test_load_skips_expired_entries(tmp_path):
    path = str(tmp_path / "cache.json")
    c1 = ModelCache(path=path, max_entries=4, ttl_s=0.05)
    c1.remember("llm", "p", "V")
    c1.save()
    time.sleep(0.08)
    c2 = ModelCache.load(path=path, max_entries=4, ttl_s=0.05)
    assert c2.size == 0


def test_load_corrupt_file_does_not_crash(tmp_path):
    path = str(tmp_path / "cache.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("this is not valid json")
    c = ModelCache.load(path=path)
    assert c.size == 0


def test_load_wrong_schema_version_ignored(tmp_path):
    path = str(tmp_path / "cache.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": 9999, "entries": []}, f)
    c = ModelCache.load(path=path)
    assert c.size == 0


def test_save_is_atomic(tmp_path):
    path = str(tmp_path / "cache.json")
    c = ModelCache(path=path, max_entries=4, ttl_s=60.0)
    c.remember("llm", "p", "V")
    c.save()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["version"] == OFFLINE_CACHE_SCHEMA_VERSION
    assert len(data["entries"]) == 1


def test_close_persists(tmp_path):
    path = str(tmp_path / "cache.json")
    c = ModelCache(path=path, max_entries=4, ttl_s=60.0)
    c.remember("llm", "p", "V")
    c.close()
    assert os.path.exists(path)


# ── Validation ───────────────────────────────────────────────────────────────


def test_invalid_max_entries_raises():
    with pytest.raises(ValueError):
        ModelCache(max_entries=0)


def test_invalid_ttl_raises():
    with pytest.raises(ValueError):
        ModelCache(ttl_s=0)


def test_clear_removes_entries(cache):
    cache.remember("llm", "p", "V")
    cache.clear()
    assert cache.size == 0
