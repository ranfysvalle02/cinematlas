"""OpenTelemetry spans: an ingest with one child per stage, and one span per search."""

import shutil

import pytest

pytest.importorskip("opentelemetry.sdk")
from opentelemetry import trace  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter  # noqa: E402

EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
trace.set_tracer_provider(_provider)  # once per process, as OpenTelemetry requires


@pytest.fixture
def spans():
    EXPORTER.clear()
    yield EXPORTER
    EXPORTER.clear()


def test_ingest_emits_a_span_per_stage_under_one_root(engine, colour_video, monkeypatch, spans):
    monkeypatch.setattr(engine, "_download_and_extract_media",
                        lambda url, d: (shutil.copy(colour_video, f"{d}/in.mp4"), None))
    monkeypatch.setattr(engine, "_transcribe_audio_safe", lambda _p: [])
    result = engine.ingest("www.b.com/v.mp4", video_id="talk")
    done = {s.name: s for s in spans.get_finished_spans()}
    root = done["cinematlas.ingest"]
    stages = [f"cinematlas.ingest.{n}" for n in ("fetched", "scenes", "transcribed", "embedded", "stored")]
    assert all(done[n].parent.span_id == root.context.span_id for n in stages)
    assert root.attributes["cinematlas.video_id"] == "talk" and root.attributes["cinematlas.scenes"] == result.scenes
    assert root.attributes["cinematlas.voyage.calls"] >= 1
    assert done["cinematlas.ingest.scenes"].attributes["cinematlas.count"] == 3
    assert all(done[a].end_time <= done[b].start_time for a, b in zip(stages, stages[1:], strict=False))  # in order


def test_each_search_is_one_span_and_cached_reads_add_none(engine, spans):
    q = engine.search("anything").only("visual").limit(2)
    q.run()
    q.run()
    searches = [s for s in spans.get_finished_spans() if s.name == "cinematlas.search"]
    assert len(searches) == 1
    assert searches[0].attributes["cinematlas.kind"] == "Search" and searches[0].attributes["cinematlas.limit"] == 2
