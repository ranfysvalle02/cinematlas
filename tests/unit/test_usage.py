"""Voyage usage metering and retry backoff."""

from types import SimpleNamespace

import pytest

from cinematlas import Cinematlas, MeteredVoyage, Usage
from cinematlas.usage import backoff, is_rate_limit, metered


class CountingVoyage:
    """Answers like Voyage, including the usage fields real responses carry."""

    def multimodal_embed(self, inputs, model, input_type):
        return SimpleNamespace(embeddings=[[0.1]] * len(inputs), text_tokens=5 * len(inputs),
                               image_pixels=1000 * len(inputs), total_tokens=7 * len(inputs))

    def embed(self, texts, model, input_type):
        return SimpleNamespace(embeddings=[[0.1]] * len(texts), total_tokens=3 * len(texts))

    def rerank(self, query, documents, model):
        return SimpleNamespace(results=[], total_tokens=11)

    extra = "passthrough"


def test_metered_client_sums_usage_per_model():
    vo = MeteredVoyage(CountingVoyage())
    vo.multimodal_embed([["a"], ["b"]], model="mm", input_type="document")
    vo.multimodal_embed([["c"]], model="mm", input_type="query")
    vo.embed(["x", "y", "z"], model="txt", input_type="document")
    vo.rerank("q", ["d1", "d2"], model="rr")
    mm, txt = vo.usage.models["mm"], vo.usage.models["txt"]
    assert (mm.calls, mm.inputs, mm.text_tokens, mm.image_pixels, mm.total_tokens) == (2, 3, 15, 3000, 21)
    assert (txt.calls, txt.total_tokens) == (1, 9)
    assert vo.usage.calls == 4 and vo.usage.total_tokens == 21 + 9 + 11
    assert vo.extra == "passthrough"  # everything else reaches the real client


def test_responses_without_usage_fields_still_count_calls():
    usage = Usage()
    usage.record("m", 4, SimpleNamespace(embeddings=[]))
    assert usage.models["m"].calls == 1 and usage.models["m"].total_tokens == 0


def test_since_reports_only_the_delta():
    vo = MeteredVoyage(CountingVoyage())
    vo.embed(["a"], model="txt", input_type="query")
    before = vo.usage.snapshot()
    vo.multimodal_embed([["a"]], model="mm", input_type="document")
    assert vo.usage.since(before) == {"mm": {"calls": 1, "inputs": 1, "text_tokens": 5,
                                             "image_pixels": 1000, "total_tokens": 7}}


def test_cost_uses_caller_prices():
    vo = MeteredVoyage(CountingVoyage())
    vo.multimodal_embed([["a"]] * 1000, model="mm", input_type="document")  # 5k text tokens, 1M pixels
    vo.embed(["t"] * 1000, model="txt", input_type="document")  # 3k tokens
    cost = vo.usage.cost({"mm": {"per_m_tokens": 0.2, "per_b_pixels": 1.0}, "txt": {"per_m_tokens": 0.1}})
    assert cost == pytest.approx(5000 / 1e6 * 0.2 + 1e6 / 1e9 * 1.0 + 3000 / 1e6 * 0.1)
    assert vo.usage.cost({}) == 0.0


def test_metered_is_idempotent():
    vo = metered(CountingVoyage())
    assert metered(vo) is vo


def test_backoff_waits_longer_on_rate_limits_and_is_capped():
    for attempt in (1, 2, 3):
        transient = backoff(attempt, RuntimeError("connection reset"))
        limited = backoff(attempt, RuntimeError("429 Too Many Requests"))
        assert 0.75 * 1.5**attempt <= transient <= 1.25 * 1.5**attempt
        assert 0.75 * (2**attempt + 1) <= limited <= 1.25 * (2**attempt + 1)
    assert backoff(20, RuntimeError("429")) == 30.0


def test_voyage_rate_limit_error_is_recognised():
    from voyageai.error import RateLimitError
    assert is_rate_limit(RateLimitError("slow down"))
    assert not is_rate_limit(ValueError("bad input"))


def test_engine_meters_its_voyage_client(fake_mongo):
    eng = Cinematlas(mongo_client=fake_mongo, voyage_client=CountingVoyage(), ping=False)
    assert isinstance(eng.vo, MeteredVoyage) and eng.usage is eng.vo.usage
    eng._embedder.text_query("hello")
    assert eng.usage.models[eng.text_model].calls == 1
