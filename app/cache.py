"""Redis-backed cache with cache-stampede / thundering-herd protection.

The problem this solves: with several API replicas behind the same Redis
instance, a cache miss on a popular key can make every replica recompute the
same expensive result at once (extra Qdrant/LLM load, and -- if the
recompute writes back inconsistently -- a race where the last writer wins
with stale data). The fix is a short-lived distributed lock around the
"miss" path: only one caller recomputes and populates the cache; everyone
else either waits briefly for it or falls back to a direct computation if
the lock isn't released in time, so a stalled worker can't wedge every
other replica.
"""
from __future__ import annotations

import json
import time
import uuid
from typing import Callable, TypeVar

from redis import Redis

from app.config import settings

T = TypeVar("T")

_LOCK_POLL_INTERVAL_SECONDS = 0.05


class Cache:
    def __init__(self, client: Redis | None = None):
        self.client = client or Redis.from_url(settings.redis_url, decode_responses=True)

    def get_or_set(
        self,
        key: str,
        compute: Callable[[], T],
        serialize: Callable[[T], str] = json.dumps,
        deserialize: Callable[[str], T] = json.loads,
        ttl_seconds: int | None = None,
        lock_timeout_seconds: int | None = None,
    ) -> T:
        ttl = ttl_seconds if ttl_seconds is not None else settings.cache_ttl_seconds
        lock_timeout = lock_timeout_seconds if lock_timeout_seconds is not None else settings.cache_lock_timeout_seconds

        cached = self.client.get(key)
        if cached is not None:
            return deserialize(cached)

        lock_key = f"lock:{key}"
        token = str(uuid.uuid4())
        acquired = self.client.set(lock_key, token, nx=True, ex=lock_timeout)

        if acquired:
            try:
                value = compute()
                self.client.set(key, serialize(value), ex=ttl)
                return value
            finally:
                # Only release if we still own it -- avoids releasing a lock
                # that expired and was re-acquired by someone else.
                self._release_if_owner(lock_key, token)

        # Someone else is computing it. Wait briefly for them to populate the
        # cache rather than recomputing in parallel; if they don't finish in
        # time, compute locally so one slow replica can't block the rest.
        deadline = time.monotonic() + lock_timeout
        while time.monotonic() < deadline:
            cached = self.client.get(key)
            if cached is not None:
                return deserialize(cached)
            time.sleep(_LOCK_POLL_INTERVAL_SECONDS)

        return compute()

    def _release_if_owner(self, lock_key: str, token: str) -> None:
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        self.client.eval(script, 1, lock_key, token)
