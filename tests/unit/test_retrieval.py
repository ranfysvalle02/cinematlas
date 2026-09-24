"""Ranking primitives: RRF fusion, sentence rerank -> moments, lexical fallback."""

import pytest

from cinematlas.retrieval import RRF_K, lexical_moment, rerank_segments, rrf_fuse

A, B, C = ("v", 0), ("v", 1), ("v", 2)


def test_rrf_rewards_agreement_across_lists():
    fused = dict(rrf_fuse({"visual": [A, B], "transcript": [B, C]}))
    assert fused[B] == pytest.approx(1 / (RRF_K + 2) + 1 / (RRF_K + 1))  # in both lists
    assert max(fused, key=fused.get) == B


def test_rrf_uses_ranks_only_so_scales_never_need_normalizing():
    # Identical rankings -> identical fusion, whatever the underlying scores were.
    assert rrf_fuse({"x": [A, B, C]}) == rrf_fuse({"y": [A, B, C]})


def test_rrf_weights_and_zero_weight_disable_a_list():
    assert rrf_fuse({"visual": [A], "transcript": [B]}, weights={"transcript": 2.0})[0][0] == B
    assert [k for k, _ in rrf_fuse({"visual": [A], "transcript": [B]}, weights={"visual": 0})] == [B]


def test_rrf_dedupes_within_a_list_and_is_deterministic_on_ties():
    fused = rrf_fuse({"x": [A, A, B]})
    assert [k for k, _ in fused] == [A, B]
    assert [k for k, _ in rrf_fuse({"x": [B], "y": [A]})] == [A, B]  # tie -> best rank, then key


def doc(scene_id, *segments, transcript=None):
    return {"video_id": "v", "scene_id": scene_id, "timestamp_start": 0.0, "timestamp_end": 10.0,
            "segments": [{"start": s, "end": s + 2, "text": t} for s, t in segments],
            "transcript": transcript or " ".join(t for _, t in segments)}


def test_rerank_scores_scene_by_its_best_sentence_and_returns_that_moment():
    docs = [doc(0, (1.0, "we test engines"), (5.0, "the boom is 110 decibels loud")),
            doc(1, (12.0, "I love hiking"))]

    def rerank_fn(query, texts):
        return [(i, 0.9 if "decibels" in t else 0.1) for i, t in enumerate(texts)]

    order, best = rerank_segments(rerank_fn, "how loud", docs)
    assert order == [A, B]
    assert best[A][1]["start"] == 5.0 and best[A][0] == 0.9


def test_rerank_falls_back_to_whole_transcript_when_segments_are_absent():
    legacy = {"video_id": "v", "scene_id": 0, "timestamp_start": 3.0, "timestamp_end": 9.0, "transcript": "old doc"}
    order, best = rerank_segments(lambda q, t: [(0, 0.5)], "q", [legacy])
    assert order == [A] and best[A][1]["start"] == 3.0


def test_rerank_skips_silent_scenes_and_caps_payload():
    calls = []
    docs = [doc(0), doc(1, *[(float(i), f"s{i}") for i in range(50)])]
    rerank_segments(lambda q, t: calls.append(t) or [], "q", docs, max_segments=10)
    assert len(calls[0]) == 10


def test_lexical_moment_picks_overlap_and_ignores_stopwords():
    segs = [{"start": 0, "text": "the and of it"}, {"start": 4, "text": "sonic boom decibels"}]
    assert lexical_moment("how loud is the sonic boom", segs)["start"] == 4
    assert lexical_moment("the", segs) is None
    assert lexical_moment("anything", []) is None


def test_to_context_is_numbered_citable_and_points_at_the_moment():
    from cinematlas import to_context

    results = [
        {"video_id": "zu", "moment": {"start": 65.2, "text": "I was born in China"},
         "moment_link": "https://cdn/v.mp4#t=65", "transcript": "whole scene"},
        {"video_id": "zu", "moment": None, "timestamp_start": 3725.0, "transcript": "", "filename": "talk.mp4"},
    ]
    assert to_context(results) == (
        "[1] zu @ 1:05 https://cdn/v.mp4#t=65\nI was born in China\n\n"
        "[2] zu @ 1:02:05 talk.mp4\n(no speech in this scene)"
    )
    assert "whole scene" in to_context(results, full_scene=True)
    assert to_context([]) == ""
