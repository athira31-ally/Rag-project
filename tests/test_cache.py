"""Verifies the cache-stampede protection: with N concurrent misses on the
same key, the expensive compute function runs exactly once.
"""
import threading

import fakeredis

from app.cache import Cache


def test_get_or_set_returns_cached_value_on_hit():
    client = fakeredis.FakeRedis(decode_responses=True)
    cache = Cache(client=client)
    calls = []

    def compute():
        calls.append(1)
        return {"value": 42}

    first = cache.get_or_set("k", compute, ttl_seconds=60, lock_timeout_seconds=5)
    second = cache.get_or_set("k", compute, ttl_seconds=60, lock_timeout_seconds=5)

    assert first == {"value": 42}
    assert second == {"value": 42}
    assert len(calls) == 1


def test_concurrent_misses_compute_only_once():
    client = fakeredis.FakeRedis(decode_responses=True)
    cache = Cache(client=client)
    call_count = 0
    lock = threading.Lock()

    def compute():
        nonlocal call_count
        with lock:
            call_count += 1
        return {"value": "expensive-result"}

    results = []

    def worker():
        results.append(cache.get_or_set("stampede-key", compute, ttl_seconds=60, lock_timeout_seconds=5))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert call_count == 1, "compute() should only run once across all concurrent callers"
    assert all(r == {"value": "expensive-result"} for r in results)
