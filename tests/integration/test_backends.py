"""The same user-facing scenarios, run against both deployment types.

Atlas (cloud) -> transcripts searched via autoEmbed.
Atlas Local   -> autoEmbed is rejected, engine falls back to client-side voyage-4 vectors.
If the product promise ("works with or without autoEmbed") breaks, one of these fails.
"""

import shutil
from types import SimpleNamespace

import pytest
from support import REAL_CLIP_SECONDS, REAL_CLIP_SHA256, REAL_CLIP_TOPICS, as_list, eventually, write_colour_video


def test_backend_resolves_to_the_expected_transcript_mode(request, backend):
    eng, mode = backend
    expected = "autoembed" if "atlas_cloud" in request.node.callspec.id else "client"
    assert mode == expected
    assert eng.transcript_mode == expected
    names = {ix["name"] for ix in eng.collection.list_search_indexes()}
    assert "cinematlas_vector_index" in names
    assert ("cinematlas_auto_index" if expected == "autoembed" else "cinematlas_transcript_index") in names


def test_transcript_search_ranks_by_meaning_not_keywords(backend, run_id, cleanup):
    eng, mode = backend
    cleanup(eng)
    texts = [
        "The chef dices onions and garlic for the sauce.",
        "Ignition. The booster clears the tower and heads to orbit.",
        "The retriever sprints across the park after a frisbee.",
    ]
    docs = [{"video_id": run_id, "scene_id": i, "transcript": t} for i, t in enumerate(texts)]
    if mode == "client":
        for d, vec in zip(docs, [eng._store_vector(v) for v in eng._embed_transcripts(texts)], strict=True):
            d["transcript_embedding"] = vec
    eng.collection.insert_many(docs)

    # No lexical overlap with the target transcript: only semantics can find it.
    results = eventually(
        lambda: (r := eng.search_transcript("space exploration", top_k=3, video_id=run_id)) and len(r) == 3 and r
    )
    assert results, f"{mode}: search never returned all 3 documents"
    assert results[0]["scene_id"] == 1
    assert results[0]["score"] > results[1]["score"]
    assert "_id" not in results[0] and "transcript_embedding" not in results[0]


def test_video_id_filter_isolates_results(backend, run_id, cleanup):
    eng, mode = backend
    cleanup(eng)
    text = "a quiet library reading room"
    vec = eng._store_vector(eng._embed_transcripts([text])[0]) if mode == "client" else None
    for vid in (run_id, f"{run_id}-other"):
        doc = {"video_id": vid, "scene_id": 0, "transcript": text}
        if vec:
            doc["transcript_embedding"] = vec
        eng.collection.insert_one(doc)

    results = eventually(lambda: eng.search_transcript("library", video_id=f"{run_id}-other"))
    assert results and {r["video_id"] for r in results} == {f"{run_id}-other"}


def test_synthetic_video_visual_and_spoken_search(backend, run_id, cleanup, tmp_path, monkeypatch):
    """Scene detection + Voyage multimodal vectors, with speech substituted for determinism."""
    eng, mode = backend
    cleanup(eng)
    video = write_colour_video(tmp_path / "v.mp4", [("red", 1.5), ("white", 1.5), ("blue", 1.5)])
    monkeypatch.setattr(eng, "_download_and_extract_media", lambda url, d: (shutil.copy(video, f"{d}/in.mp4"), None))
    monkeypatch.setattr(eng, "_transcribe_audio_safe", lambda _p: [
        {"start": 0.1, "end": 1.4, "text": "Fresh strawberries and tomatoes at the farmers market."},
        {"start": 1.6, "end": 2.9, "text": "Snow falls silently over the frozen mountain pass."},
        {"start": 3.1, "end": 4.4, "text": "Waves crash against the hull far out at sea."},
    ])

    assert eng.ingest_video("https://example.com/v.mp4", video_id=run_id) == 3
    stored = list(eng.collection.find({"video_id": run_id}))
    assert all(len(as_list(d["visual_embedding"])) == 1024 for d in stored)
    if mode == "client":
        assert all(len(as_list(d["transcript_embedding"])) == 1024 for d in stored)
    else:
        assert all("transcript_embedding" not in d for d in stored)

    visual = eventually(lambda: (r := eng.search_visual_vector(
        "a solid bright red image", top_k=3, video_id=run_id)) and len(r) == 3 and r)
    assert visual and visual[0]["scene_id"] == 0
    assert eng.search_visual_vector("a solid blue image", top_k=3, video_id=run_id)[0]["scene_id"] == 2

    spoken = eventually(lambda: (r := eng.search_transcript(
        "winter weather", top_k=3, video_id=run_id)) and len(r) == 3 and r)
    assert spoken and spoken[0]["scene_id"] == 1
    assert spoken[0]["deep_link"] == "https://example.com/v.mp4#t=1"


def _assert_real_clip_indexed(eng, vid):
    docs = list(eng.collection.find({"video_id": vid}).sort("scene_id", 1))
    assert len(docs) == 4, "fixture has exactly 3 built-in cuts + 1 source cut"
    assert docs[-1]["timestamp_end"] == pytest.approx(REAL_CLIP_SECONDS, abs=0.5)
    assert all(d["visual_embedding"] and len(as_list(d["visual_embedding"])) == 1024 for d in docs)
    transcripts = [d["transcript"].lower() for d in docs]
    assert all(p in transcripts[0] for p in REAL_CLIP_TOPICS["aircraft"])
    assert all(p in transcripts[1] for p in REAL_CLIP_TOPICS["outdoors"])
    assert all(p in transcripts[-1] for p in REAL_CLIP_TOPICS["beer"])
    return docs


def _assert_real_clip_searchable(eng, vid):
    # Each question must come back as the *scene* that answers it, through Atlas, on real speech.
    def top(query):
        hits = eventually(lambda: (r := eng.search_transcript(query, top_k=4, video_id=vid)) and len(r) >= 3 and r)
        assert hits, f"no results for {query!r}"
        return hits[0]

    # The outdoors segment spans a real source cut, so it legitimately owns scenes 1 and 2.
    assert top("outdoor sports and recreation")["scene_id"] in (1, 2)
    assert top("brewing craft beer")["scene_id"] == 3
    assert top("wind tunnel testing of a supersonic plane")["scene_id"] == 0

    # Hybrid search: fused sources, reranked sentences, and a deep link to the exact moment.
    hits = eventually(lambda: (r := eng.search("how many medals has the beer won?", top_k=3, video_id=vid))
                      and "rerank" in r[0]["ranks"] and r)
    assert hits and hits[0]["scene_id"] == 3
    assert "medals" in hits[0]["moment"]["text"].lower()
    assert hits[0]["moment"]["start"] > hits[0]["timestamp_start"], "moment is inside the scene, not its start"
    assert set(hits[0]["ranks"]) >= {"visual", "scene", "transcript", "rerank"}

    visual = eventually(lambda: eng.search_visual_vector("a person being interviewed", top_k=4, video_id=vid))
    assert visual and {r["video_id"] for r in visual} == {vid}
    assert [r["score"] for r in visual] == sorted((r["score"] for r in visual), reverse=True)


@pytest.mark.media
def test_real_url_end_to_end(backend, run_id, cleanup, test_whisper_model, clip_server, monkeypatch):
    """Nothing substituted: yt-dlp (HTTP) -> ffmpeg -> faster-whisper -> Voyage -> Atlas -> search."""
    pytest.importorskip("faster_whisper")
    eng, _ = backend
    cleanup(eng)
    eng.whisper_model = test_whisper_model
    monkeypatch.setattr(eng, "allow_private_urls", True)  # the fixture server is on loopback
    url = f"{clip_server}/x59_quiet_crew.mp4"

    assert eng.ingest_video(url, video_id=run_id) == 4
    docs = _assert_real_clip_indexed(eng, run_id)
    assert {d["source_type"] for d in docs} == {"url"}
    assert docs[1]["deep_link"] == f"{url}#t={int(docs[1]['timestamp_start'])}"
    _assert_real_clip_searchable(eng, run_id)


@pytest.mark.media
def test_real_file_upload_end_to_end(backend, run_id, cleanup, test_whisper_model, real_clip):
    """The upload entrypoint with a real file object, as a web handler would pass it."""
    pytest.importorskip("faster_whisper")
    eng, _ = backend
    cleanup(eng)
    eng.whisper_model = test_whisper_model

    with open(real_clip, "rb") as fh:
        upload = SimpleNamespace(file=fh, filename="quiet-crew.mp4")  # FastAPI UploadFile shape
        assert eng.ingest_file(upload, video_id=run_id) == 4
    docs = _assert_real_clip_indexed(eng, run_id)
    assert {d["source_type"] for d in docs} == {"file"}
    assert {d["filename"] for d in docs} == {"quiet-crew.mp4"}
    assert {d["content_sha256"] for d in docs} == {REAL_CLIP_SHA256}
    assert {d["deep_link"] for d in docs} == {None}
    _assert_real_clip_searchable(eng, run_id)


def test_doctor_reports_a_healthy_deployment_with_graceful_warnings(backend, run_id, cleanup):
    """Real clusters: no failures; anything unsupported is a warning with a fix, never an error."""
    eng, mode = backend
    cleanup(eng)
    eng.collection.insert_one({"video_id": run_id, "scene_id": 0, "status": "COMPLETED", "transcript": "x",
                               "segments": [], "scene_embedding": None})
    report = eng.doctor()
    by_name = {c.name: c for c in report.checks}
    print(report)  # visible with -s: the exact report a user sees
    assert report.ok, str(report)
    assert by_name["MongoDB"].status == "ok" and by_name["Voyage AI"].status == "ok"
    assert all(by_name[f"Index {n}"].status in ("ok", "warn") for n in eng.desired_indexes())
    assert by_name["$rerank"].status in ("ok", "warn") and all(c.fix for c in report.problems)
    assert eng.native_fusion is (by_name["$rankFusion"].status == "ok")


def test_ensure_indexes_update_is_idempotent_on_a_real_cluster(backend):
    eng, mode = backend
    assert eng.ensure_indexes(update=True) == mode
    assert all(s.ok for s in eng.inspect_indexes())
