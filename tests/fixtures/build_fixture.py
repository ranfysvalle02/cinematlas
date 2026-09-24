"""Rebuild tests/fixtures/x59_quiet_crew.mp4 from the public-domain NASA source.

Source: "NASA's The Quiet Crew | Joe Dussling" (NASA HQ, 2022-01-07, nasa_id NHQ20220107ARMD)
        https://images.nasa.gov/details/NHQ20220107ARMD
License: NASA media is not copyrighted (public domain) - https://www.nasa.gov/nasa-brand-center/images-and-media/

Three sentence-aligned segments on distinct topics are joined with hard cuts, giving
ground truth for scene alignment: each scene's speech lies entirely inside that scene.

    uv run python tests/fixtures/build_fixture.py
"""

import hashlib
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

SOURCE_URL = "https://images-assets.nasa.gov/video/NHQ20220107ARMD/NHQ20220107ARMD~small.mp4"
OUT = Path(__file__).with_name("x59_quiet_crew.mp4")
# (start, end) in source seconds, cut on sentence boundaries.
SEGMENTS = [
    (30.5, 52.6),   # scene 0: X-59 aerodynamics, engine distortion, sonic boom
    (74.9, 84.5),   # scene 1: hiking, kayaking, running, Ultimate Frisbee, disc golf
    (102.6, 110.6), # scene 2: roasted jalapeno beer, competition medals
]


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "source.mp4"
        print(f"downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, src)

        chains, labels = [], []
        for i, (a, b) in enumerate(SEGMENTS):
            chains.append(f"[0:v]trim={a}:{b},setpts=PTS-STARTPTS[v{i}];"
                          f"[0:a]atrim={a}:{b},asetpts=PTS-STARTPTS[a{i}]")
            labels.append(f"[v{i}][a{i}]")
        graph = ";".join(chains) + ";" + "".join(labels) + f"concat=n={len(SEGMENTS)}:v=1:a=1[vc][a];"
        # 240p @ 15 fps: tiny, and still plenty for scene detection + Voyage keyframes.
        graph += "[vc]scale=-2:240,fps=15[v]"

        subprocess.run([
            "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(src),
            "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
            # H.264 baseline + 16 kHz mono AAC: decodable by every OpenCV/ffmpeg build.
            "-c:v", "libx264", "-profile:v", "baseline", "-crf", "30",
            "-preset", "veryslow", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ac", "1", "-ar", "16000", "-b:a", "32k",
            "-movflags", "+faststart", "-map_metadata", "-1", str(OUT),
        ], check=True)

    digest = hashlib.sha256(OUT.read_bytes()).hexdigest()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KiB) sha256={digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
