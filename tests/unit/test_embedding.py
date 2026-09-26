"""Batched keyframe embedding: alignment, resilience, and memory hygiene.

The single most important invariant: output[i] is the embedding of scenes[i].
If that slips, every downstream document silently carries the wrong vector.
"""

import pytest
from PIL import Image
from support import COLOURS_BGR, as_list

INDEX = {name: float(i) for i, name in enumerate(COLOURS_BGR)}


def img(colour):
    b, g, r = COLOURS_BGR[colour]
    return Image.new("RGB", (8, 8), (r, g, b))


def scenes(*colours):
    return [{"scene_id": i, "keyframe_image": img(c) if c else None} for i, c in enumerate(colours)]


def test_one_embedding_per_scene_in_order(engine):
    out = engine._embed_keyframes_batched(scenes("red", "green", "blue"), batch_size=2)
    assert out == [[INDEX["red"]], [INDEX["green"]], [INDEX["blue"]]]


def test_scenes_without_keyframes_stay_aligned(engine, fake_voyage):
    # Regression: missing frames (keyframe_image=None) were either sent to Voyage as None
    # or dropped, shifting every later vector onto the wrong scene.
    out = engine._embed_keyframes_batched(scenes("red", None, "blue", None, "green"), batch_size=16)
    assert out == [[INDEX["red"]], None, [INDEX["blue"]], None, [INDEX["green"]]]
    assert fake_voyage.calls[0]["colours"] == ["red", "blue", "green"]


def test_batch_with_no_keyframes_makes_no_api_call(engine, fake_voyage):
    assert engine._embed_keyframes_batched(scenes(None, None), batch_size=2) == [None, None]
    assert fake_voyage.calls == []


def test_batches_respect_batch_size_and_use_document_input_type(engine, fake_voyage):
    engine._embed_keyframes_batched(scenes(*["red"] * 5), batch_size=2)
    assert [len(c["inputs"]) for c in fake_voyage.calls] == [2, 2, 1]
    assert {c["input_type"] for c in fake_voyage.calls} == {"document"}
    assert {c["model"] for c in fake_voyage.calls} == {"voyage-multimodal-3.5"}


def test_transient_failure_is_retried(engine, fake_voyage):
    fake_voyage.failures = 2
    out = engine._embed_keyframes_batched(scenes("red"), max_retries=3)
    assert out == [[INDEX["red"]]]
    assert len(fake_voyage.calls) == 3


def test_exhausted_retries_null_only_that_batch(engine, fake_voyage):
    fake_voyage.failures = 2  # first batch burns both attempts; second batch succeeds
    out = engine._embed_keyframes_batched(scenes("red", "green", "blue"), batch_size=2, max_retries=2)
    assert out == [None, None, [INDEX["blue"]]]


def test_short_response_is_treated_as_failure_not_misaligned(engine, fake_voyage):
    fake_voyage.drop_one = True
    out = engine._embed_keyframes_batched(scenes("red", "green"), max_retries=2)
    assert out == [None, None]
    assert len(fake_voyage.calls) == 2


def test_images_are_closed_and_released_after_each_batch(engine):
    batch = scenes("red", "green")
    images = [s["keyframe_image"] for s in batch]
    engine._embed_keyframes_batched(batch)
    assert all("keyframe_image" not in s for s in batch)
    for im in images:
        with pytest.raises(ValueError):  # PIL raises on use after close()
            im.load()


def test_images_released_even_when_embedding_fails(engine, fake_voyage):
    fake_voyage.failures = 99
    batch = scenes("red")
    engine._embed_keyframes_batched(batch, max_retries=1)
    assert "keyframe_image" not in batch[0]


def test_rejects_non_positive_batch_size(engine):
    with pytest.raises(ValueError):
        engine._embed_keyframes_batched(scenes("red"), batch_size=0)


def test_joint_scene_vectors_embed_image_with_its_own_transcript(engine, fake_voyage):
    batch = scenes("red", "white", "blue")
    engine._embed_keyframes_batched(batch, transcripts=["red words", "", "blue words"])
    joint_call = fake_voyage.calls[1]
    assert [len(i) for i in joint_call["inputs"]] == [2, 2]  # [image, text] interleaved
    assert [i[1] for i in joint_call["inputs"]] == ["red words", "blue words"]  # silent scene skipped
    assert "scene_embedding" in batch[0] and "scene_embedding" not in batch[1]


def test_ingest_stores_joint_vectors_and_silent_scenes_reuse_keyframe_vector(engine, fake_mongo, colour_video,
                                                                              monkeypatch):
    import shutil

    monkeypatch.setattr(engine, "_download_and_extract_media",
                        lambda u, d: (shutil.copy(colour_video, f"{d}/in.mp4"), None))
    monkeypatch.setattr(engine, "_transcribe_audio_safe", lambda _p: [{"start": 0.2, "end": 1.2, "text": "hi"}])
    engine.ingest("https://youtu.be/abcdefghijk")
    docs = sorted(fake_mongo.collection.docs, key=lambda d: d["scene_id"])
    assert docs[0]["scene_embedding"] is not None
    assert as_list(docs[1]["scene_embedding"]) == as_list(docs[1]["visual_embedding"])  # no speech -> no extra call
    assert docs[0]["segments"] == [{"start": 0.2, "end": 1.2, "text": "hi"}]


def test_scene_embeddings_can_be_disabled(engine, fake_voyage):
    engine.scene_embeddings = False
    batch = scenes("red")
    engine._embed_keyframes_batched(batch)
    assert len(fake_voyage.calls) == 1 and "scene_embedding" not in batch[0]
