# SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
# SPDX-FileCopyrightText: 2026 Fernando Rodriguez
"""Brain Offline Cache (B03).

Cache for VLM/LLM responses + model manifest so the brain can keep
operating when LM Studio or internet connectivity is reduced or absent.

Features:
  * SHA256-keyed response cache (prompt + optional image bytes / context).
  * Prompt deduplication — identical recent prompts return the cached
    VLM/LLM response instead of hitting the model server again.
  * Local "safe" fallback responses classified by request kind when the
    remote model is unreachable.
  * JSON persistence — cache is loaded at startup and flushed on
    ``save()`` / ``close()``.
  * Size-limited LRU eviction with TTL.
  * Stats: hit rate, miss count, eviction count, fallback count, size.

Usage (explicit, opt-in — existing call sites are not touched):

    cache = ModelCache.load("data/model_cache.json")

    key = cache.make_key("vlm", prompt, image_bytes=img)
    cached = cache.get(key)
    if cached is not None:
        return cached
    response = vlm_call(...)
    cache.put(key, response)

Or via the decorator:

    @cache.cached(kind="llm")
    def decide(scene: str, sensors: dict, task: str) -> str:
        ...

The cache can also serve a canned fallback response when the remote model
is unreachable:

    try:
        resp = vlm_call(...)
    except Exception:
        resp = cache.fallback("vlm", classification="unknown")

Rules followed:
  * No magic numbers — every numeric value is a named constant or config.
  * Python 3.10+ only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger("brain.offline_cache")


# ---------------------------------------------------------------------------
# Constants — no magic numbers
# ---------------------------------------------------------------------------

## Maximum number of entries held in the cache before LRU eviction kicks in.
OFFLINE_CACHE_MAX_ENTRIES: int = 1024

## Default time-to-live for a cache entry, in seconds.
OFFLINE_CACHE_TTL_S: float = 24.0 * 60.0 * 60.0  # 24 hours

## Default persistence path.
OFFLINE_CACHE_DEFAULT_PATH: str = "data/model_cache.json"

## Current on-disk schema version. Bump when the serialised layout changes.
OFFLINE_CACHE_SCHEMA_VERSION: int = 1

## Hash algorithm used to derive cache keys.
OFFLINE_CACHE_HASH_ALGO: str = "sha256"

## Length (in hex chars) of a truncated digest used for human-readable keys.
OFFLINE_CACHE_KEY_HEX_LEN: int = 32  # 128-bit prefix of SHA-256

## Fallback classifications — what the brain should do when the model is
## unreachable and no cached response exists. The string values are the
## canned responses served to the caller.
FALLBACK_RESPONSES: dict[str, dict[str, str]] = {
    # kind -> classification -> canned response
    "llm": {
        "unknown": "INVESTIGATE forward",
        "clear": "FORWARD 30",
        "obstacle": "STOP",
        "low_power": "STOP",
        "default": "STOP",
    },
    "vlm": {
        "unknown": "Scene unclear. Possible obstacles ahead. Proceed with caution.",
        "clear": "Path appears clear. No obvious obstacles detected.",
        "obstacle": "Obstacle detected ahead. Stop and re-evaluate.",
        "default": "Scene unavailable (model offline). Assume obstacles present.",
    },
    "task": {
        "unknown": '[{"skill": "STOP", "args": {}}]',
        "default": '[{"skill": "STOP", "args": {}}]',
    },
}

## The default classification used by ``fallback()`` if none is supplied.
FALLBACK_DEFAULT_CLASSIFICATION: str = "default"

## Minimum ratio (0..1) used when computing hit rate, guarding div-by-zero.
HIT_RATE_MIN_DENOM: int = 1


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """One cached response."""

    key: str
    kind: str  # "vlm" | "llm" | "task" | ...
    response: str
    created_at: float
    last_used: float
    hits: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def is_expired(self, now: float, ttl_s: float) -> bool:
        return (now - self.created_at) > ttl_s


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Runtime cache statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    fallbacks: int = 0
    writes: int = 0

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total < HIT_RATE_MIN_DENOM:
            return 0.0
        return self.hits / total

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hit_rate"] = self.hit_rate()
        return d


# ---------------------------------------------------------------------------
# ModelCache
# ---------------------------------------------------------------------------


class ModelCache:
    """LRU + TTL cache for VLM/LLM responses, with local fallbacks.

    Thread-safe. Persisted to a JSON file.
    """

    def __init__(
        self,
        path: str = OFFLINE_CACHE_DEFAULT_PATH,
        max_entries: int = OFFLINE_CACHE_MAX_ENTRIES,
        ttl_s: float = OFFLINE_CACHE_TTL_S,
        fallbacks: Optional[dict[str, dict[str, str]]] = None,
    ):
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        if ttl_s <= 0:
            raise ValueError("ttl_s must be > 0")

        self._path = path
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._fallbacks = fallbacks if fallbacks is not None else FALLBACK_RESPONSES
        self._entries: "OrderedDict[str, CacheEntry]" = OrderedDict()
        self._stats = CacheStats()
        self._lock = threading.RLock()
        self._model_manifest: dict[str, Any] = {}

    # -- construction ---------------------------------------------------------

    @classmethod
    def load(
        cls,
        path: str = OFFLINE_CACHE_DEFAULT_PATH,
        max_entries: int = OFFLINE_CACHE_MAX_ENTRIES,
        ttl_s: float = OFFLINE_CACHE_TTL_S,
    ) -> "ModelCache":
        """Instantiate a cache, loading from disk if the file exists."""
        inst = cls(path=path, max_entries=max_entries, ttl_s=ttl_s)
        inst._load_from_disk()
        return inst

    # -- keys -----------------------------------------------------------------

    @staticmethod
    def hash_bytes(data: bytes) -> str:
        """Return a truncated SHA-256 hex digest of ``data``."""
        h = hashlib.new(OFFLINE_CACHE_HASH_ALGO, data).hexdigest()
        return h[:OFFLINE_CACHE_KEY_HEX_LEN]

    @classmethod
    def make_key(
        cls,
        kind: str,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        extra: Optional[str] = None,
    ) -> str:
        """Build a deterministic cache key for the given request."""
        hasher = hashlib.new(OFFLINE_CACHE_HASH_ALGO)
        hasher.update(kind.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(prompt.encode("utf-8"))
        if image_bytes is not None:
            hasher.update(b"\x00")
            hasher.update(hashlib.new(OFFLINE_CACHE_HASH_ALGO, image_bytes).digest())
        if extra is not None:
            hasher.update(b"\x00")
            hasher.update(extra.encode("utf-8"))
        return hasher.hexdigest()[:OFFLINE_CACHE_KEY_HEX_LEN]

    # -- get / put ------------------------------------------------------------

    def get(self, key: str) -> Optional[str]:
        """Return the cached response for ``key`` or ``None``.

        Expired entries are evicted transparently.
        """
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.is_expired(now, self._ttl_s):
                del self._entries[key]
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            entry.last_used = now
            entry.hits += 1
            self._entries.move_to_end(key)
            self._stats.hits += 1
            return entry.response

    def put(
        self,
        key: str,
        response: str,
        kind: str = "unknown",
        meta: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert or refresh a cache entry."""
        now = time.time()
        with self._lock:
            if key in self._entries:
                entry = self._entries[key]
                entry.response = response
                entry.last_used = now
                entry.created_at = now
                entry.kind = kind
                if meta is not None:
                    entry.meta = dict(meta)
                self._entries.move_to_end(key)
            else:
                entry = CacheEntry(
                    key=key,
                    kind=kind,
                    response=response,
                    created_at=now,
                    last_used=now,
                    hits=0,
                    meta=dict(meta) if meta else {},
                )
                self._entries[key] = entry
            self._stats.writes += 1
            self._evict_if_needed()

    def remember(
        self,
        kind: str,
        prompt: str,
        response: str,
        image_bytes: Optional[bytes] = None,
        extra: Optional[str] = None,
    ) -> str:
        """Convenience: build a key from (prompt, image) and store ``response``.

        Returns the key used. Useful for callers that prefer not to deal
        with keys explicitly.
        """
        key = self.make_key(kind, prompt, image_bytes=image_bytes, extra=extra)
        self.put(key, response, kind=kind)
        return key

    def lookup(
        self,
        kind: str,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        extra: Optional[str] = None,
    ) -> Optional[str]:
        """Convenience: build a key and call :meth:`get`."""
        key = self.make_key(kind, prompt, image_bytes=image_bytes, extra=extra)
        return self.get(key)

    # -- fallback -------------------------------------------------------------

    def fallback(
        self,
        kind: str,
        classification: str = FALLBACK_DEFAULT_CLASSIFICATION,
    ) -> str:
        """Return a local canned response for ``kind`` / ``classification``.

        Falls through classifications -> the kind's "default" -> a
        hard-coded last-resort string.
        """
        with self._lock:
            self._stats.fallbacks += 1
        by_kind = self._fallbacks.get(kind, {})
        if classification in by_kind:
            return by_kind[classification]
        if FALLBACK_DEFAULT_CLASSIFICATION in by_kind:
            return by_kind[FALLBACK_DEFAULT_CLASSIFICATION]
        # Last-resort safe string — not from FALLBACK_RESPONSES.
        return "STOP"

    # -- decorator ------------------------------------------------------------

    def cached(
        self,
        kind: str,
        key_from_args: Optional[Callable[..., str]] = None,
    ) -> Callable:
        """Decorator factory that caches a function's return value.

        By default the key is derived from ``repr((args, kwargs))``.
        Pass ``key_from_args`` to override (it must return the prompt
        string, not the full key).
        """

        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any):
                if key_from_args is not None:
                    prompt = key_from_args(*args, **kwargs)
                else:
                    prompt = repr((args, kwargs))
                key = self.make_key(kind, prompt)
                hit = self.get(key)
                if hit is not None:
                    return hit
                result = fn(*args, **kwargs)
                # Only cache string-like results; other shapes are
                # round-tripped via JSON to preserve structure.
                if isinstance(result, str):
                    self.put(key, result, kind=kind)
                else:
                    try:
                        self.put(key, json.dumps(result), kind=kind)
                    except (TypeError, ValueError):
                        # Non-serialisable — skip caching, return as-is.
                        return result
                return result

            return wrapper

        return decorator

    # -- model manifest -------------------------------------------------------

    def set_model_manifest(self, manifest: dict[str, Any]) -> None:
        """Record the current model manifest (names, hashes, versions)."""
        with self._lock:
            self._model_manifest = dict(manifest)

    def get_model_manifest(self) -> dict[str, Any]:
        """Return a copy of the current model manifest."""
        with self._lock:
            return dict(self._model_manifest)

    # -- stats / sizes --------------------------------------------------------

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._entries)

    @property
    def max_entries(self) -> int:
        return self._max_entries

    @property
    def ttl_s(self) -> float:
        return self._ttl_s

    def stats(self) -> CacheStats:
        """Return a snapshot of the cache statistics (thread-safe copy)."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                expirations=self._stats.expirations,
                fallbacks=self._stats.fallbacks,
                writes=self._stats.writes,
            )

    def reset_stats(self) -> None:
        with self._lock:
            self._stats = CacheStats()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    # -- eviction -------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        # Must be called while holding the lock.
        while len(self._entries) > self._max_entries:
            # OrderedDict FIFO order == LRU because put/get move_to_end.
            self._entries.popitem(last=False)
            self._stats.evictions += 1

    def purge_expired(self) -> int:
        """Proactively drop expired entries. Returns how many were removed."""
        now = time.time()
        removed = 0
        with self._lock:
            for key in list(self._entries.keys()):
                if self._entries[key].is_expired(now, self._ttl_s):
                    del self._entries[key]
                    removed += 1
                    self._stats.expirations += 1
        return removed

    # -- persistence ----------------------------------------------------------

    def _load_from_disk(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not load cache from %s: %s", self._path, e)
            return

        version = data.get("version")
        if version != OFFLINE_CACHE_SCHEMA_VERSION:
            logger.warning(
                "Cache schema mismatch (have=%s, want=%s). Ignoring on-disk cache.",
                version,
                OFFLINE_CACHE_SCHEMA_VERSION,
            )
            return

        entries_raw = data.get("entries", [])
        manifest = data.get("model_manifest", {})
        now = time.time()
        loaded = 0
        skipped = 0
        with self._lock:
            self._model_manifest = dict(manifest) if isinstance(manifest, dict) else {}
            for raw in entries_raw:
                try:
                    entry = CacheEntry(
                        key=raw["key"],
                        kind=raw.get("kind", "unknown"),
                        response=raw["response"],
                        created_at=float(raw.get("created_at", now)),
                        last_used=float(raw.get("last_used", now)),
                        hits=int(raw.get("hits", 0)),
                        meta=dict(raw.get("meta", {})),
                    )
                except (KeyError, TypeError, ValueError):
                    skipped += 1
                    continue
                if entry.is_expired(now, self._ttl_s):
                    skipped += 1
                    continue
                self._entries[entry.key] = entry
                loaded += 1
            self._evict_if_needed()
        logger.info("Loaded %d cache entries from %s (%d skipped)", loaded, self._path, skipped)

    def save(self, path: Optional[str] = None) -> None:
        """Flush the cache to disk as JSON."""
        target = path or self._path
        with self._lock:
            payload = {
                "version": OFFLINE_CACHE_SCHEMA_VERSION,
                "saved_at": time.time(),
                "model_manifest": dict(self._model_manifest),
                "entries": [asdict(e) for e in self._entries.values()],
            }
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp = target + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, target)
        logger.info("Saved %d cache entries to %s", len(payload["entries"]), target)

    def close(self) -> None:
        """Persist and release the cache."""
        try:
            self.save()
        except OSError as e:
            logger.warning("Failed to save cache on close: %s", e)
