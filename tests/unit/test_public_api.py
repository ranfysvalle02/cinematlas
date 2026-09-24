"""Guard the published surface: what users import must keep working."""

import cinematlas
from cinematlas import auto_embed_index_definition, visual_index_definition


def test_public_names_are_importable():
    for name in cinematlas.__all__:
        assert hasattr(cinematlas, name), name


def test_version_is_resolved_from_package_metadata():
    assert cinematlas.__version__ != "0.0.0"


def test_index_definitions_cover_every_field_the_queries_use():
    visual = {f["path"]: f for f in visual_index_definition(512)["fields"]}
    assert visual["visual_embedding"]["numDimensions"] == 512
    assert visual["video_id"]["type"] == "filter"  # required by the video_id search filter

    auto = {f["path"]: f for f in auto_embed_index_definition("voyage-4-lite")["fields"]}
    assert auto["transcript"] == {"type": "autoEmbed", "modality": "text", "path": "transcript",
                                  "model": "voyage-4-lite"}
    assert auto["video_id"]["type"] == "filter"


def test_importing_the_package_does_not_import_heavy_video_stack():
    import subprocess
    import sys

    code = "import sys, cinematlas; print(any(m in sys.modules for m in ('cv2', 'yt_dlp', 'scenedetect')))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_vector_indexes_use_scalar_quantization_by_default():
    from cinematlas.indexes import scene_index_definition, text_index_definition

    for definition in (visual_index_definition(), scene_index_definition()):
        assert definition["fields"][0]["quantization"] == "scalar"
    assert "quantization" not in visual_index_definition(quantization=None)["fields"][0]
    assert text_index_definition()["mappings"]["fields"]["video_id"] == {"type": "token"}


def test_readme_links_work_on_pypi():
    """README.md is the PyPI long description; relative links 404 there, so every link must be absolute."""
    import re
    from pathlib import Path

    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text()
    relative = [t for t in re.findall(r"\]\(([^)\s]+)\)", readme) if not t.startswith(("http://", "https://", "#"))]
    assert relative == [], f"use absolute GitHub URLs in README.md: {relative}"
