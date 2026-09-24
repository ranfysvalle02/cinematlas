"""Benchmark corpus: six NASA "The Quiet Crew" interviews (public domain).

Deliberately hard: one program, one topic (the X-59 quiet supersonic aircraft),
overlapping vocabulary. Retrieval must separate *which person* said *what*, *when*.
"""

EPISODES = {
    "durston": "NHQ20210805ARMD01",
    "blomquist": "NHQ20230310ARMD01",
    "cliatt": "NHQ20220811ARMD01",
    "titus": "NHQ20220915ARMD02",
    "zu": "NHQ20230504ARMD01",
    "watters": "NHQ20221110ARMD01",
}


def url(nasa_id: str) -> str:
    return f"https://images-assets.nasa.gov/video/{nasa_id}/{nasa_id}~small.mp4"


DB = "cinematlas_bench"
COLLECTIONS = {"autoembed": "scenes_autoembed", "client": "scenes_client"}

# Caption ablation: the same corpus with the burned-in caption band cropped off every keyframe
# before embedding. Transcripts are untouched. Captions sit in the bottom ~13% of each frame.
NO_CAPTIONS = "scenes_nocaptions"
CAPTION_BAND = 0.15
