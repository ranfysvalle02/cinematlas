"""Pure-function contracts. These encode the invariants the rest of the system relies on."""

from urllib.parse import parse_qs, urlparse

import pytest

from cinematlas._utils import (
    MAX_NUM_CANDIDATES,
    assign_segments,
    assign_transcripts,
    build_deep_link,
    build_vector_search_pipeline,
    extract_video_id,
    normalize_segments,
    split_long_spans,
)


class TestExtractVideoId:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtube.com/watch?v=dQw4w9WgXcQ&t=42s",
            "https://m.youtube.com/watch?feature=share&v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ?si=abc",
            "https://www.youtube.com/shorts/dQw4w9WgXcQ",
            "https://www.youtube.com/embed/dQw4w9WgXcQ",
            "https://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ",
        ],
    )
    def test_every_youtube_form_maps_to_the_canonical_id(self, url):
        assert extract_video_id(url) == "dQw4w9WgXcQ"

    def test_non_youtube_ids_are_deterministic(self):
        url = "https://cdn.example.com/media/talk.mp4"
        assert extract_video_id(url) == extract_video_id(url)
        assert extract_video_id(url).startswith("vid_")

    def test_distinct_non_youtube_urls_never_collide(self):
        # Regression: an 11-char regex fallback mapped both of these to "episode_001",
        # and since ingestion does delete_many({video_id}), ingesting one wiped the other.
        a = "https://cdn.example.com/episode_001_part_a.mp4"
        b = "https://cdn.example.com/episode_001_part_b.mp4"
        assert extract_video_id(a) != extract_video_id(b)

    def test_youtube_host_without_id_falls_back_to_hash(self):
        assert extract_video_id("https://www.youtube.com/feed/trending").startswith("vid_")


class TestBuildDeepLink:
    def test_appends_timestamp_to_query_string(self):
        assert build_deep_link("https://www.youtube.com/watch?v=abc", 12.9) == (
            "https://www.youtube.com/watch?v=abc&t=12s"
        )

    def test_url_without_query_gets_a_valid_query(self):
        # Regression: naive f"{url}&t=" produced "https://youtu.be/abc&t=5s" (broken link).
        assert build_deep_link("https://youtu.be/abc", 5) == "https://youtu.be/abc?t=5s"

    def test_existing_timestamp_is_replaced_not_duplicated(self):
        link = build_deep_link("https://www.youtube.com/watch?v=abc&t=99s&list=x", 3)
        assert parse_qs(urlparse(link).query) == {"v": ["abc"], "t": ["3s"], "list": ["x"]}


class TestAssignTranscripts:
    SEGMENTS = [
        {"start": 0.0, "end": 2.0, "text": "hello"},
        {"start": 2.0, "end": 4.0, "text": "world"},
        {"start": 3.5, "end": 6.5, "text": "spanning"},
        {"start": 7.0, "end": 8.0, "text": ""},
    ]

    def test_segment_touching_a_boundary_is_not_duplicated(self):
        assert assign_transcripts([(0.0, 2.0), (2.0, 4.0)], self.SEGMENTS[:2]) == ["hello", "world"]

    def test_segment_spanning_multiple_scenes_appears_in_each(self):
        out = assign_transcripts([(2.0, 5.0), (5.0, 7.0)], self.SEGMENTS)
        assert out == ["world spanning", "spanning"]

    def test_timestamp_jitter_across_a_cut_does_not_duplicate_the_sentence(self):
        # Real case from the NASA fixture: a sentence ends 0.07s after the cut.
        segs = [{"start": 18.0, "end": 22.2, "text": "sonic boom"}, {"start": 22.2, "end": 24.8, "text": "hiking"}]
        assert assign_transcripts([(0.0, 22.13), (22.13, 29.2)], segs) == ["sonic boom", "hiking"]

    def test_short_segment_is_never_dropped(self):
        # Entirely inside a sliver of overlap below the threshold: still goes to its best scene.
        segs = [{"start": 9.9, "end": 10.2, "text": "yes"}]
        assert assign_transcripts([(0.0, 10.0), (10.0, 20.0)], segs) == ["", "yes"]

    def test_segment_outside_every_scene_is_ignored(self):
        assert assign_transcripts([(0.0, 1.0)], [{"start": 5.0, "end": 6.0, "text": "late"}]) == [""]

    def test_scene_without_speech_and_blank_segments_yield_empty_string(self):
        assert assign_transcripts([(7.0, 9.0)], self.SEGMENTS) == [""]

    def test_output_is_aligned_one_to_one_with_scenes(self):
        spans = [(0, 1), (10, 11), (1, 2)]
        assert len(assign_transcripts(spans, self.SEGMENTS)) == len(spans)


def test_normalize_segments_accepts_dicts_sdk_objects_and_none():
    class Seg:
        start, end, text = 1, 2.5, "  obj  "

    out = normalize_segments([{"start": 0, "end": 1, "text": " d "}, Seg(), {"text": None}])
    assert out == [
        {"start": 0.0, "end": 1.0, "text": "d"},
        {"start": 1.0, "end": 2.5, "text": "obj"},
        {"start": 0.0, "end": 0.0, "text": ""},
    ]
    assert normalize_segments(None) == []


class TestVectorSearchPipeline:
    def test_text_query_targets_autoembed(self):
        (stage, project) = build_vector_search_pipeline(index_name="ix", path="transcript", top_k=3, query_text="q")
        assert stage["$vectorSearch"] == {
            "index": "ix", "path": "transcript", "query": "q", "numCandidates": 30, "limit": 3,
        }
        assert project["$project"]["score"] == {"$meta": "vectorSearchScore"}
        assert project["$project"]["_id"] == 0

    def test_vector_query_with_video_filter(self):
        (stage, _) = build_vector_search_pipeline(
            index_name="ix", path="visual_embedding", top_k=1, query_vector=[0.1], match={"video_id": "v1"}
        )
        assert stage["$vectorSearch"]["queryVector"] == [0.1]
        assert stage["$vectorSearch"]["filter"] == {"video_id": "v1"}
        assert "query" not in stage["$vectorSearch"]

    def test_num_candidates_is_capped_at_atlas_limit(self):
        (stage, _) = build_vector_search_pipeline(index_name="i", path="p", top_k=5000, query_text="q")
        assert stage["$vectorSearch"]["numCandidates"] == MAX_NUM_CANDIDATES

    @pytest.mark.parametrize("top_k", [0, -1, 2.5, None])
    def test_rejects_invalid_top_k(self, top_k):
        with pytest.raises(ValueError):
            build_vector_search_pipeline(index_name="i", path="p", top_k=top_k, query_text="q")

    @pytest.mark.parametrize("kwargs", [{}, {"query_text": "q", "query_vector": [1.0]}])
    def test_requires_exactly_one_query_kind(self, kwargs):
        with pytest.raises(ValueError):
            build_vector_search_pipeline(index_name="i", path="p", top_k=1, **kwargs)

    def test_returned_projection_is_a_copy(self):
        (_, p1) = build_vector_search_pipeline(index_name="i", path="p", top_k=1, query_text="q")
        p1["$project"]["leak"] = 1
        (_, p2) = build_vector_search_pipeline(index_name="i", path="p", top_k=1, query_text="q")
        assert "leak" not in p2["$project"]


class TestSplitLongSpans:
    def test_short_spans_are_untouched(self):
        assert split_long_spans([(0.0, 10.0), (10.0, 25.0)], 30) == [(0.0, 10.0), (10.0, 25.0)]

    def test_long_span_splits_into_equal_contiguous_parts(self):
        assert split_long_spans([(10.0, 100.0)], 30) == [(10.0, 40.0), (40.0, 70.0), (70.0, 100.0)]

    def test_uneven_length_never_exceeds_max(self):
        parts = split_long_spans([(0.0, 61.0)], 30)
        assert len(parts) == 3 and all(b - a <= 30 for a, b in parts) and parts[-1][1] == 61.0

    def test_exact_multiple_does_not_create_a_sliver(self):
        assert split_long_spans([(0.0, 60.0)], 30) == [(0.0, 30.0), (30.0, 60.0)]

    def test_disabled(self):
        assert split_long_spans([(0.0, 600.0)], None) == [(0.0, 600.0)]


class TestWordLevelAssignment:
    """Real word timings from the NASA fixture around the cut at 22.13s."""

    CUT = [(0.0, 22.13), (22.13, 29.2)]

    def w(self, *triples):
        return [{"start": a, "end": b, "word": t} for a, b, t in triples]

    def test_grazing_segment_goes_whole_to_its_scene_even_if_first_word_jitters_early(self):
        # "I" is stamped 21.80-22.26 (midpoint before the cut) but the sentence belongs after it.
        seg = {"start": 21.80, "end": 24.20, "text": "I love hiking.",
               "words": self.w((21.80, 22.26, " I"), (22.26, 22.42, " love"), (22.42, 22.70, " hiking."))}
        out = assign_segments(self.CUT, [seg])
        assert out[0] == [] and [s["text"] for s in out[1]] == ["I love hiking."]

    def test_straddling_segment_is_split_at_the_cut_instead_of_duplicated(self):
        seg = {"start": 20.0, "end": 24.0, "text": "sonic boom. I love hiking.",
               "words": self.w((20.0, 20.8, " sonic"), (20.9, 21.5, " boom."),
                               (22.5, 23.0, " I"), (23.0, 23.4, " love"), (23.4, 24.0, " hiking."))}
        out = assign_segments(self.CUT, [seg])
        assert out[0] == [{"start": 20.0, "end": 21.5, "text": "sonic boom."}]
        assert out[1] == [{"start": 22.5, "end": 24.0, "text": "I love hiking."}]

    def test_straddling_segment_without_words_is_kept_in_both(self):
        seg = {"start": 20.0, "end": 24.0, "text": "no word timings"}
        out = assign_segments(self.CUT, [seg])
        assert [s["text"] for s in out[0]] == [s["text"] for s in out[1]] == ["no word timings"]

    def test_segments_are_time_ordered_per_scene(self):
        segs = [{"start": 5.0, "end": 6.0, "text": "b"}, {"start": 1.0, "end": 2.0, "text": "a"}]
        assert [s["text"] for s in assign_segments([(0.0, 10.0)], segs)[0]] == ["a", "b"]


def test_normalize_segments_keeps_word_timestamps():
    class W:
        def __init__(self, s, e, t):
            self.start, self.end, self.word = s, e, t

    class Seg:
        start, end, text = 0.0, 1.0, " hi there "
        words = [W(0.0, 0.4, " hi"), W(0.5, 1.0, " there")]

    (out,) = normalize_segments([Seg()])
    assert out["words"] == [{"start": 0.0, "end": 0.4, "word": " hi"}, {"start": 0.5, "end": 1.0, "word": " there"}]
