"""Query-embedding cache: correct LRU semantics, and at engine level, fewer API calls and identical results."""

import threading

import pytest

from cinematlas._cache import LRUCache


def test_hits_misses_and_lru_eviction():
    cache, calls = LRUCache(2), []

    def compute(v):
        return lambda: calls.append(v) or v

    assert cache.get_or_compute("a", compute(1)) == 1
    assert cache.get_or_compute("a", compute(99)) == 1  # hit: compute not called
    cache.get_or_compute("b", compute(2))
    cache.get_or_compute("a", compute(99))  # touch "a" so "b" is least recent
    cache.get_or_compute("c", compute(3))  # evicts "b"
    assert cache.get_or_compute("b", compute(22)) == 22
    assert calls == [1, 2, 3, 22]
    assert cache.info() == {"hits": 2, "misses": 4, "size": 2, "maxsize": 2}


def test_failures_are_never_cached():
    cache, attempts = LRUCache(4), []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("429")
        return "ok"

    with pytest.raises(RuntimeError):
        cache.get_or_compute("q", flaky)
    assert cache.get_or_compute("q", flaky) == "ok" and len(attempts) == 2


def test_size_zero_disables_caching():
    cache, calls = LRUCache(0), []
    for _ in range(3):
        cache.get_or_compute("q", lambda: calls.append(1))
    assert len(calls) == 3 and cache.info()["size"] == 0


def test_negative_size_is_rejected():
    with pytest.raises(ValueError):
        LRUCache(-1)


def test_concurrent_access_is_safe():
    cache = LRUCache(50)
    errors = []

    def worker(i):
        try:
            for j in range(200):
                assert cache.get_or_compute((i + j) % 80, lambda k=(i + j) % 80: k) == (i + j) % 80
        except Exception as e:  # pragma: no cover - only on failure
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and cache.info()["size"] <= 50


# ---------------------------------------------------------------- engine level
@pytest.fixture
def indexed(engine, fake_mongo):
    engine._resolved_transcript_mode = "client"
    scene = {"video_id": "v", "scene_id": 0, "transcript": "sonic boom", "segments": [], "video_url": None}
    fake_mongo.collection.results_by_index = {name: [scene] for name in (
        "cinematlas_vector_index", "cinematlas_scene_index", "cinematlas_transcript_index")}
    return engine


def test_repeated_searches_skip_voyage_and_return_identical_results(indexed, fake_voyage):
    first = indexed.search("sonic boom", rerank=False)
    embeds_after_first = [c for c in fake_voyage.calls if "colours" in c or "texts" in c]
    second = indexed.search("sonic boom", rerank=False)
    embeds_after_second = [c for c in fake_voyage.calls if "colours" in c or "texts" in c]
    assert second == first
    assert len(embeds_after_first) == 2 and len(embeds_after_second) == 2  # multimodal + text, once each
    assert indexed.query_cache_info()["hits"] == 2


def test_cache_keys_are_exact_and_model_specific(indexed, fake_voyage):
    indexed.search_visual_vector("Sonic boom")
    indexed.search_visual_vector("sonic boom")  # different text -> different key
    indexed.model = "voyage-multimodal-4"
    indexed.search_visual_vector("sonic boom")  # different model -> different key
    assert len([c for c in fake_voyage.calls if "colours" in c]) == 3


def test_cache_can_be_disabled_and_cleared(fake_mongo, fake_voyage):
    from cinematlas import Cinematlas

    eng = Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False, query_cache_size=0)
    eng.search_visual_vector("q")
    eng.search_visual_vector("q")
    assert len(fake_voyage.calls) == 2

    eng2 = Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False)
    eng2.search_visual_vector("q")
    eng2.clear_query_cache()
    eng2.search_visual_vector("q")
    assert len(fake_voyage.calls) == 4 and eng2.query_cache_info()["misses"] == 1
