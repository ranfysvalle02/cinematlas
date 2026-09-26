"""Hybrid search orchestration: fusion across sources, moments, explainability, degradation."""

import pytest
from pymongo.errors import OperationFailure

from cinematlas import SearchError

URL = "https://cdn.example.com/talk.mp4"


def scene(scene_id, *segments, start=None):
    segs = [{"start": s, "end": s + 3, "text": t} for s, t in segments]
    return {"video_id": "v", "scene_id": scene_id, "video_url": URL,
            "timestamp_start": start if start is not None else (segs[0]["start"] if segs else 0.0),
            "timestamp_end": 30.0, "transcript": " ".join(t for _, t in segments), "segments": segs,
            "deep_link": f"{URL}#t=0", "score": 0.5}


AIR = scene(0, (0.0, "we tested the aerodynamics"), (9.0, "the sonic boom measured 110 decibels"))
HIKE = scene(1, (31.0, "I love hiking and kayaking"))
SILENT = scene(2, start=60.0)


@pytest.fixture
def indexed(engine, fake_mongo):
    engine._resolved_transcript_mode = "client"
    fake_mongo.collection.results_by_index = {
        "cinematlas_vector_index": [SILENT, HIKE, AIR],  # visual thinks the silent scene looks best
        "cinematlas_scene_index": [AIR, HIKE],
        "cinematlas_transcript_index": [AIR, HIKE],
    }
    return engine


def test_fuses_sources_and_returns_the_moment_not_the_scene_start(indexed):
    (top, *_rest) = indexed.search("how many decibels was the sonic boom").limit(3).routing("adaptive").run()
    assert top["scene_id"] == 0
    assert top["moment"] == {"start": 9.0, "end": 12.0, "text": "the sonic boom measured 110 decibels"}
    assert top["moment_link"] == f"{URL}#t=9"  # deep link to the second it is said
    assert top["ranks"] == {"visual": 3, "scene": 1, "transcript": 1, "rerank": 1}
    assert top["relevance"] > 0


def test_visual_question_routes_to_visual_lists_and_finds_silent_scenes(indexed):
    # No transcript sentence matches -> reranker relevance ~0 -> "about what's shown".
    results = indexed.search("jet silhouette against grey clouds").limit(3).routing("adaptive").run()
    assert results.speech_confidence == 0.0
    silent = next(r for r in results if r["scene_id"] == 2)  # only the visual lists know it
    assert silent["moment"] is None and silent["ranks"] == {"visual": 1}
    assert results.weights == {"visual": 0.25, "scene": 1.0, "transcript": 0.0, "text": 0.0, "rerank": 0.0}


def test_speech_question_routes_to_speech_lists(indexed):
    results = indexed.search("sonic boom measured decibels").limit(3).routing("adaptive").run()
    assert results.speech_confidence == 1.0
    assert results[0]["scene_id"] == 0 and "decibels" in results[0]["moment"]["text"]
    assert 2 not in [r["scene_id"] for r in results], "visual-only scenes don't compete with confident speech"


def test_fixed_routing_keeps_classic_fusion(indexed):
    results = indexed.search("sonic boom").limit(3).routing("fixed").run()
    assert results.speech_confidence is None and 2 in [r["scene_id"] for r in results]


def test_explicit_weights_disable_routing(indexed):
    results = indexed.search("sonic boom").limit(3).weights(**{"visual": 1}).run()
    assert results.speech_confidence is None and results.weights["visual"] == 1


def test_invalid_routing_is_rejected(indexed):
    with pytest.raises(ValueError):
        indexed.search("q").routing("magic").run()


def test_default_is_scene_first_ranking_with_the_reranker_picking_the_moment(indexed):
    results = indexed.search("how many decibels was the sonic boom").limit(3).run()
    assert [r["scene_id"] for r in results] == [0, 1]  # the joint vector's order, untouched by the reranker
    assert results[0]["moment"]["text"] == "the sonic boom measured 110 decibels"
    assert results[0]["moment_link"] == f"{URL}#t=9"
    assert results.speech_confidence is None
    assert all(set(r["ranks"]) <= {"scene", "rerank"} for r in results)


def test_default_falls_back_to_fusion_when_scenes_have_no_joint_vectors(indexed, fake_mongo):
    fake_mongo.collection.results_by_index["cinematlas_scene_index"] = []  # ingested before scene vectors
    results = indexed.search("sonic boom decibels").run()
    assert results and results.speech_confidence is not None


def test_weights_or_sources_without_routing_keep_fusion(indexed):
    assert "transcript" in indexed.search("sonic boom").weights(**{"transcript": 1})[0]["ranks"]
    assert indexed.search("sonic boom decibels").using(*("scene", "transcript")).speech_confidence is not None


def test_one_multimodal_query_embedding_serves_visual_and_scene(indexed, fake_voyage):
    indexed.search("sonic boom").run()
    assert len([c for c in fake_voyage.calls if "colours" in c]) == 1


def test_sources_can_be_restricted(indexed, fake_mongo):
    indexed.search("sonic boom").using(*("transcript",)).rerank(False).run()
    queried = {p[0]["$vectorSearch"]["index"] for p in fake_mongo.collection.pipelines if "$vectorSearch" in p[0]}
    assert queried == {"cinematlas_transcript_index"}


def test_without_rerank_moment_comes_from_lexical_overlap(indexed, fake_voyage):
    (top, *_r) = indexed.search("sonic boom decibels").rerank(False).routing("adaptive").run()
    assert top["moment"]["start"] == 9.0 and top["relevance"] is None
    assert not [c for c in fake_voyage.calls if "rerank" in c]


def test_reranker_outage_degrades_instead_of_failing(indexed, fake_voyage):
    fake_voyage.rerank_error = RuntimeError("503")
    results = indexed.search("sonic boom decibels").run()
    assert results and all("rerank" not in r["ranks"] for r in results)


def test_missing_scene_index_is_skipped(indexed, fake_mongo):
    # e.g. a collection ingested before scene embeddings existed
    fake_mongo.collection.index_errors["cinematlas_scene_index"] = OperationFailure("index not found")
    results = indexed.search("sonic boom").run()
    assert results and all("scene" not in r["ranks"] for r in results)


def test_all_sources_failing_raises(indexed, fake_mongo):
    fake_mongo.collection.aggregate_error = OperationFailure("cluster down")
    with pytest.raises(SearchError, match="All search sources failed"):
        indexed.search("sonic boom").run()


def test_uploads_have_no_moment_link(indexed, fake_mongo):
    for docs in fake_mongo.collection.results_by_index.values():
        for d in docs:
            d["video_url"] = None
    assert all(r["moment_link"] is None for r in indexed.search("sonic boom"))


@pytest.mark.parametrize("build", [lambda q: q.limit(0), lambda q: q.using("audio"), lambda q: q.only("audio")])
def test_invalid_arguments_fail_when_building_the_query(indexed, build):
    with pytest.raises(ValueError):
        build(indexed.search("q"))


def test_empty_query_short_circuits(indexed, fake_mongo):
    assert indexed.search("") == [] and fake_mongo.collection.pipelines == []


# ---------------------------------------------------------------- Atlas-native execution
def fusion_row(doc, **ranks):
    return {**{k: v for k, v in doc.items() if k != "score"}, "score": 0.05,
            "fusion": {"details": [{"inputPipelineName": n, "rank": r} for n, r in ranks.items()]
                       + [{"inputPipelineName": "text", "rank": "NA"}]}}


def test_native_rank_fusion_is_one_round_trip_with_the_same_ranking(indexed, fake_mongo):
    python_path = [r["scene_id"] for r in indexed.search("sonic boom decibels").rerank(False).routing("adaptive")]

    fake_mongo.collection.pipelines.clear()
    fake_mongo.collection.native_fusion_rows = [
        fusion_row(AIR, visual=3, scene=1, transcript=1), fusion_row(HIKE, visual=2, scene=2, transcript=2),
        fusion_row(SILENT, visual=1)]
    indexed.native_fusion = None  # re-detect
    native = indexed.search("sonic boom decibels").rerank(False).routing("adaptive").run()

    assert [r["scene_id"] for r in native] == python_path, "native and client-side fusion must agree"
    assert indexed.native_fusion is True
    (stage,) = [p[0]["$rankFusion"] for p in fake_mongo.collection.pipelines if "$rankFusion" in p[0]]
    assert set(stage["input"]["pipelines"]) == {"visual", "scene", "transcript", "text"}
    assert stage["combination"]["weights"]["visual"] == 0.25 and stage["scoreDetails"] is True
    assert native[0]["ranks"] == {"visual": 3, "scene": 1, "transcript": 1}  # "NA" ranks ignored


def test_unsupported_rank_fusion_is_detected_once_then_skipped(indexed, fake_mongo):
    indexed.search("sonic boom").run()
    indexed.search("sonic boom").run()
    attempts = [p for p in fake_mongo.collection.pipelines if "$rankFusion" in p[0]]
    assert len(attempts) == 1 and indexed.native_fusion is False


def test_native_rerank_scores_sentences_server_side(indexed, fake_mongo, fake_voyage):
    fake_mongo.collection.native_rerank_rows = [
        {"video_id": "v", "scene_id": 1, "segment": HIKE["segments"][0], "relevance": 0.2},
        {"video_id": "v", "scene_id": 0, "segment": AIR["segments"][1], "relevance": 0.9},
        {"video_id": "v", "scene_id": 0, "segment": AIR["segments"][0], "relevance": 0.1},
    ]
    (top, *_r) = indexed.search("how loud was it").run()
    assert top["scene_id"] == 0 and top["moment"]["start"] == 9.0 and top["relevance"] == 0.9
    assert not [c for c in fake_voyage.calls if "rerank" in c], "no client-side rerank call"
    (pipe,) = [p for p in fake_mongo.collection.pipelines if any("$rerank" in st for st in p)]
    assert {"$unwind": {"path": "$segments", "preserveNullAndEmptyArrays": True}} in pipe


def test_disabled_native_rerank_falls_back_to_voyage_api(indexed, fake_voyage):
    indexed.search("sonic boom decibels").run()
    assert indexed.native_rerank is False
    assert [c for c in fake_voyage.calls if "rerank" in c]


def test_text_source_uses_atlas_search_with_video_filter(indexed, fake_mongo):
    indexed.search("X-59 building 4826").using(*("text",)).video("v").rerank(False).run()
    (stage,) = [p[0]["$search"] for p in fake_mongo.collection.pipelines if "$search" in p[0]]
    assert stage["index"] == "cinematlas_text_index"
    assert stage["compound"]["filter"] == [{"equals": {"path": "video_id", "value": "v"}}]


# ---------------------------------------------------------------- calibration-aware routing
def test_uncalibrated_reranker_steps_down_to_fixed_fusion_and_says_so_once(indexed, caplog, monkeypatch):
    import logging

    from cinematlas import capabilities

    monkeypatch.setattr(capabilities, "_WARNED", set())
    indexed.rerank_model = "rerank-2.5-lite"  # different score scale: no calibration on record
    with caplog.at_level(logging.WARNING, logger="cinematlas"):
        first = indexed.search("jet silhouette against grey clouds").routing("adaptive").run()
        indexed.search("sonic boom decibels").routing("adaptive").run()
    assert first.speech_confidence is None, "never route on an uncalibrated score scale"
    assert first.weights["visual"] == 0.25 and first.weights["rerank"] == 2.0  # fixed defaults
    assert [r.message for r in caplog.records].count(
        "No routing calibration for reranker 'rerank-2.5-lite'; search() uses fixed fusion. "
        "Calibrate with bench/ and pass routing_thresholds=(lo, hi), or use rerank-2.5.") == 1


def test_custom_calibration_is_honoured(indexed):
    indexed.rerank_model = "rerank-2.5-lite"
    indexed.routing_thresholds = (0.0, 0.01)  # any matching sentence counts as "about speech"
    assert indexed.search("sonic boom decibels").routing("adaptive").speech_confidence == 1.0


@pytest.mark.parametrize("bad", [(0.6, 0.4), (-0.1, 0.5), (0.2, 1.5)])
def test_invalid_calibration_is_rejected_up_front(fake_mongo, fake_voyage, bad):
    from cinematlas import Cinematlas

    with pytest.raises(ValueError, match="routing_thresholds"):
        Cinematlas(mongo_client=fake_mongo, voyage_client=fake_voyage, ping=False, routing_thresholds=bad)


@pytest.mark.parametrize(("model", "thresholds", "status", "detail"), [
    ("rerank-2.5", None, "ok", "calibrated for rerank-2.5 (0.45–0.55)"),
    ("rerank-2.5-lite", None, "warn", "no calibration for rerank-2.5-lite; using fixed fusion"),
    ("rerank-2.5-lite", (0.3, 0.4), "ok", "custom calibration (0.30–0.40)"),
    (None, None, "info", "off (no reranker)"),
])
def test_doctor_reports_routing_state(engine, model, thresholds, status, detail):
    engine.rerank_model, engine.routing_thresholds = model, thresholds
    check = next(c for c in engine.doctor(check_voyage=False).checks if c.name == "Routing")
    assert check.status == status and detail in check.detail
