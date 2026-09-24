"""The developer-facing surface: result objects, unified ingest, progress, diagnostics,
graceful native fallbacks, and self-healing indexes."""

import json
import logging
import shutil

import pytest
from pymongo.errors import OperationFailure
from support import FakeCollection

from cinematlas import IngestResult, SearchHit, SearchResults
from cinematlas import engine as engine_module
from cinematlas.doctor import explain_native_failure
from cinematlas.indexes import (
    definition_drift,
    desired_indexes,
    ensure_search_indexes,
    inspect_indexes,
    visual_index_definition,
)

URL = "https://cdn.example.com/talk.mp4"
HIT = {"video_id": "zu", "scene_id": 7, "timestamp_start": 63.0, "transcript": "whole scene text",
       "moment": {"start": 65.2, "end": 69.9, "text": "I was born in China"},
       "moment_link": f"{URL}#t=65", "ranks": {"transcript": 1, "visual": 4, "rerank": 1},
       "relevance": 0.91, "score": 0.0487}


# ---------------------------------------------------------------- result objects
class TestSearchHit:
    def test_is_still_a_plain_json_serializable_dict(self):
        hit = SearchHit(HIT)
        assert hit == HIT and isinstance(hit, dict) and json.loads(json.dumps(hit)) == HIT

    def test_attribute_access_and_shortcuts(self):
        hit = SearchHit(HIT)
        assert hit.video_id == "zu" and hit.scene_id == 7
        assert hit.start == 65.2 and hit.timestamp == "1:05"
        assert hit.text == "I was born in China" and hit.link == f"{URL}#t=65"
        with pytest.raises(AttributeError):
            _ = hit.no_such_field

    def test_falls_back_to_scene_when_there_is_no_moment(self):
        hit = SearchHit({**HIT, "moment": None, "moment_link": None, "deep_link": f"{URL}#t=63"})
        assert hit.start == 63.0 and hit.text == "whole scene text" and hit.link == f"{URL}#t=63"

    def test_explain_orders_sources_by_rank(self):
        assert SearchHit(HIT).explain() == "transcript #1, rerank #1, visual #4, relevance 0.91, score 0.0487"

    def test_repr_is_informative(self):
        assert repr(SearchHit(HIT)) == "<SearchHit zu#7 @ 1:05 'I was born in China'>"


class TestSearchResults:
    def test_str_is_a_readable_table_with_links(self):
        text = str(SearchResults([HIT]))
        assert text.splitlines() == [" 1. zu#7 @    1:05  I was born in China", f"    {URL}#t=65"]
        assert str(SearchResults()) == "(no results)"

    def test_notebook_markdown(self):
        md = SearchResults([HIT])._repr_markdown_()
        assert f"| 1 | zu#7 | 1:05 | I was born in China | [open]({URL}#t=65) |" in md

    def test_helpers(self):
        results = SearchResults([HIT, {**HIT, "scene_id": 8}])
        assert results.top.scene_id == 7 and results.links == [f"{URL}#t=65"] * 2
        assert results.to_context().startswith("[1] zu @ 1:05")
        assert SearchResults() == [] and SearchResults().top is None


# ---------------------------------------------------------------- unified ingest + progress
@pytest.fixture
def served(engine, colour_video, monkeypatch):
    monkeypatch.setattr(engine, "_download_and_extract_media",
                        lambda url, d: (shutil.copy(colour_video, f"{d}/in.mp4"), None))
    monkeypatch.setattr(engine, "_transcribe_audio_safe", lambda _p: [{"start": 0.2, "end": 1.2, "text": "hi"}])
    return engine


def test_ingest_url_returns_a_typed_result(served):
    result = served.ingest("www.b.com/v.mp4", video_id="talk")
    assert isinstance(result, IngestResult)
    assert (result.video_id, result.scenes, result.spoken_scenes, result.source_type) == ("talk", 3, 1, "url")
    assert set(result.stages) == {"fetched", "scenes", "transcribed", "embedded", "stored"}
    assert str(result).startswith("Indexed 3 scenes (1 with speech) as 'talk'")


@pytest.mark.parametrize("kind", ["path", "bytes", "fileobj"])
def test_ingest_accepts_files_in_every_shape(served, colour_video, kind, monkeypatch):
    monkeypatch.setattr(served, "_download_video", lambda *a, **k: pytest.fail("files must not download"))
    source = {"path": str(colour_video), "bytes": colour_video.read_bytes(),
              "fileobj": open(colour_video, "rb")}[kind]  # noqa: SIM115
    result = served.ingest(source)
    assert result.source_type == "file" and result.scenes == 3 and result.video_id.startswith("file_")


def test_progress_reports_every_stage_in_order(served):
    events = []
    served.ingest(URL, progress=lambda stage, info: events.append((stage, info)))
    assert [e[0] for e in events] == ["fetched", "scenes", "transcribed", "embedded", "stored"]
    assert dict(events)["scenes"] == {"count": 3} and dict(events)["stored"] == {"scenes": 3}


def test_engine_level_default_progress_and_broken_callbacks_are_harmless(served):
    def explode(stage, info):
        raise RuntimeError("UI crashed")

    served.progress = explode
    assert served.ingest(URL).scenes == 3


def test_legacy_methods_still_return_counts(served):
    assert served.ingest_video(URL) == 3


# ---------------------------------------------------------------- graceful native fallbacks
@pytest.mark.parametrize(("message", "reason", "fix"), [
    ("$rerank is not enabled for demo. Enable the $rerank Project Setting", "disabled for this Atlas project",
     "Project Settings"),
    ("Unrecognized pipeline stage name: '$rerank'", "needs MongoDB 8.3+", "Upgrade"),
    ("Unrecognized pipeline stage name: '$rankFusion'", "needs MongoDB 8.0+", "Upgrade"),
])
def test_native_failures_are_explained_with_a_fix(message, reason, fix):
    stage = "$rankFusion" if "rankFusion" in message else "$rerank"
    got_reason, got_fix = explain_native_failure(stage, OperationFailure(message))
    assert reason in got_reason and fix in got_fix


def test_unrecognized_stage_on_a_new_enough_server_blames_the_deployment_not_the_version():
    reason, fix = explain_native_failure("$rerank", OperationFailure("Unrecognized pipeline stage name: '$rerank'"),
                                         server_version="8.3.11")  # Atlas Local
    assert "isn't available on this deployment" in reason and "Atlas-only" in reason
    assert "needs MongoDB" not in reason


def test_search_mapping_drift_ignores_server_added_field_options():
    from cinematlas.indexes import text_index_definition

    want = text_index_definition()
    atlas_local = {"mappings": {"dynamic": False, "fields": {
        "transcript": {"type": "string", "analyzer": "lucene.english", "indexOptions": "offsets",
                       "store": True, "norms": "include"},
        "video_id": {"type": "token"}}}}
    assert definition_drift(atlas_local, want) == ()
    changed = {"mappings": {"dynamic": False, "fields": {"transcript": {"type": "string",
                                                                          "analyzer": "lucene.standard"}}}}
    assert set(definition_drift(changed, want)) == {
        "mappings.fields.transcript.analyzer: 'lucene.standard' -> 'lucene.english'",
        "mappings.fields.video_id: missing"}


def test_disabled_rerank_warns_once_with_the_fix_and_search_still_works(engine, fake_mongo, caplog, monkeypatch):
    monkeypatch.setattr(engine_module, "_WARNED", set())
    engine._resolved_transcript_mode = "client"
    fake_mongo.collection.results_by_index = {"cinematlas_transcript_index": [
        {"video_id": "v", "scene_id": 0, "transcript": "sonic boom", "segments": []}]}
    with caplog.at_level(logging.WARNING, logger="cinematlas"):
        for _ in range(3):
            engine.native_rerank = None  # fresh engines re-probe; the warning must still appear once
            assert engine.search("sonic boom")
    warnings = [r.message for r in caplog.records if "$rerank" in r.message]
    assert len(warnings) == 1
    assert "using Voyage rerank API (same model)" in warnings[0] and "Fix:" in warnings[0]


# ---------------------------------------------------------------- self-healing indexes
def test_drift_ignores_server_defaults_but_catches_changed_settings():
    want = visual_index_definition(quantization="scalar")
    server = {"fields": [{**want["fields"][0], "hnswOptions": {"maxEdges": 16}}, want["fields"][1]]}
    assert definition_drift(server, want) == ()
    old = {"fields": [{k: v for k, v in want["fields"][0].items() if k != "quantization"}, want["fields"][1]]}
    assert definition_drift(old, want) == ("visual_embedding.quantization: None -> 'scalar'",)
    assert definition_drift({"fields": [want["fields"][0]]}, want) == ("video_id: missing filter field",)


def test_inspect_reports_missing_ready_and_drifted():
    coll = FakeCollection()
    coll.search_indexes["cinematlas_vector_index"] = visual_index_definition(quantization=None)
    report = {s.name: s for s in inspect_indexes(coll, desired_indexes("client"))}
    assert report["cinematlas_vector_index"].state == "ready" and report["cinematlas_vector_index"].drift
    assert report["cinematlas_text_index"].state == "missing" and not report["cinematlas_text_index"].ok


def test_outdated_indexes_warn_by_default_and_update_in_place_on_request(caplog):
    coll = FakeCollection()
    coll.search_indexes["cinematlas_vector_index"] = visual_index_definition(quantization=None)
    with caplog.at_level(logging.WARNING, logger="cinematlas"):
        ensure_search_indexes(coll, transcript_mode="client", wait=False)
    assert "cinematlas setup --update" in caplog.text
    assert ("update_search_index", "cinematlas_vector_index") not in coll.ops

    ensure_search_indexes(coll, transcript_mode="client", wait=False, update=True)
    assert ("update_search_index", "cinematlas_vector_index") in coll.ops
    assert all(s.ok for s in inspect_indexes(coll, desired_indexes("client")))


# ---------------------------------------------------------------- doctor
def test_doctor_on_a_healthy_deployment(engine, fake_mongo):
    engine._resolved_transcript_mode = "client"
    ensure_search_indexes(fake_mongo.collection, transcript_mode="client", wait=False)
    fake_mongo.collection.native_fusion_rows = []
    fake_mongo.collection.native_rerank_rows = []
    fake_mongo.collection.docs.append({"video_id": "v", "scene_id": 0, "status": "COMPLETED", "segments": [],
                                       "scene_embedding": [0.1]})
    report = engine.doctor()
    by_name = {c.name: c for c in report.checks}
    assert report.ok, str(report)
    assert by_name["$rankFusion"].status == "ok" and engine.native_fusion is True
    assert by_name["$rerank"].status == "ok" and engine.native_rerank is True
    assert by_name["Data"].detail == "1 scenes across 1 videos"


def test_doctor_explains_every_problem_with_a_fix(engine, fake_mongo, monkeypatch):
    engine._resolved_transcript_mode = "client"
    fake_mongo.collection.search_indexes["cinematlas_vector_index"] = visual_index_definition(quantization=None)
    fake_mongo.collection.docs += [
        {"video_id": "old", "scene_id": 0, "status": "COMPLETED"},  # pre-0.2 document
        {"video_id": "bad", "status": "FAILED", "error_message": "403"},
    ]
    monkeypatch.setattr("cinematlas.engine.shutil.which", lambda _n: None)
    report = engine.doctor(check_voyage=False)
    by_name = {c.name: c for c in report.checks}

    assert not report.ok
    assert by_name["Index cinematlas_vector_index"].status == "warn"
    assert "update=True" in by_name["Index cinematlas_vector_index"].fix
    assert by_name["Index cinematlas_text_index"].status == "fail"
    assert by_name["$rerank"].status == "warn" and "Voyage rerank API" in by_name["$rerank"].detail
    assert by_name["Legacy scenes"].status == "warn" and by_name["Failed ingests"].detail.endswith(": bad")
    assert by_name["ffmpeg"].status == "fail" and all(c.fix for c in report.problems)
    assert "→" in str(report) and "problem(s)" in str(report)


def test_doctor_stops_early_when_mongodb_is_unreachable(engine, fake_mongo):
    def down(cmd):
        raise OperationFailure("connection refused")

    fake_mongo.admin.command = down
    report = engine.doctor()
    assert [c.name for c in report.checks] == ["MongoDB"] and not report.ok


def test_engine_repr_is_informative_and_does_no_io(engine, fake_mongo):
    before = list(fake_mongo.collection.ops)
    assert repr(engine) == "<Cinematlas cinematlas_enterprise.multimodal_scenes · transcripts=auto · rerank=rerank-2.5>"
    engine._resolved_transcript_mode, engine.rerank_model = "autoembed", None
    assert repr(engine).endswith("transcripts=autoembed · rerank=off>")
    assert fake_mongo.collection.ops == before and fake_mongo.collection.pipelines == []
