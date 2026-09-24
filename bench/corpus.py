"""Benchmark corpus: six NASA "The Quiet Crew" interviews (public domain).

Deliberately hard: one program, one topic (the X-59 quiet supersonic aircraft),
overlapping vocabulary. Retrieval must separate *which person* said *what*, *when*.
"""

from urllib.parse import quote

EPISODES = {
    "durston": "NHQ20210805ARMD01",
    "blomquist": "NHQ20230310ARMD01",
    "cliatt": "NHQ20220811ARMD01",
    "titus": "NHQ20220915ARMD02",
    "zu": "NHQ20230504ARMD01",
    "watters": "NHQ20221110ARMD01",
}


def url(nasa_id: str) -> str:
    safe = quote(nasa_id)  # some NASA ids contain spaces and non-ASCII
    return f"https://images-assets.nasa.gov/video/{safe}/{safe}~small.mp4"


DB = "cinematlas_bench"
COLLECTIONS = {"autoembed": "scenes_autoembed", "client": "scenes_client"}

# Caption ablation: the same corpus with the burned-in caption band cropped off every keyframe
# before embedding. Transcripts are untouched. Captions sit in the bottom ~13% of each frame.
NO_CAPTIONS = "scenes_nocaptions"
CAPTION_BAND = 0.15

# Held-out corpus: a different domain with no burned-in captions. Narrated ISS tours, astronaut Q&A,
# science demos. Questions were written by an agent that saw only these videos' keyframes and
# transcripts, never the code or any results. Routing thresholds were not tuned on it.
STATION_DB = "cinematlas_bench_station"
STATION_COLLECTION = "scenes_autoembed"
STATION = {
    "tour": "Step_Inside_the_Station-4K",
    "potty": "NHQ_2020_0925_AskNASA┃ HOW DO ASTRONAUTS USE THE POTTY IN SPACE",
    "nbl": "jsc2024m000064_WBTD-NBL-Finalized",
    "pettit": "jsc2024m000154_Astronaut Moments Don Pettit_MP4",
    "rubins": "jsc2021m000150_Kate_Rubins_Scientist_in_Space-MP4",
    "food": "SS_SpaceFoodScientist_Final_Social",
}
