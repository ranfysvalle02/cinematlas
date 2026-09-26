"""CLI: routing, output formats, exit codes. The engine is replaced by a recorder."""

import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cinematlas import Check, Diagnosis, IndexStatus, IngestionError, IngestResult, SearchResults, cli

HIT = {"video_id": "v", "scene_id": 3, "moment": {"start": 34.0, "text": "110 decibels"},
       "moment_link": "https://b.com/v.mp4#t=34", "score": 0.05}


class RecordingEngine:
    instances: list["RecordingEngine"] = []
    diagnosis = Diagnosis([Check("MongoDB", "ok", "connected")])

    def __init__(self, uri, **kwargs):
        self.uri, self.kwargs, self.calls = uri, kwargs, []
        RecordingEngine.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.calls.append(("close",))

    def ensure_indexes(self, **kw):
        self.calls.append(("ensure_indexes", kw))
        return "client"

    def inspect_indexes(self):
        return [IndexStatus("cinematlas_vector_index", "vectorSearch", "ready")]

    def doctor(self, **kw):
        self.calls.append(("doctor", kw))
        return RecordingEngine.diagnosis

    def ingest(self, source, **kw):
        self.calls.append(("ingest", source, kw))
        if kw.get("video_id") == "bad":
            raise IngestionError("Not a decodable video file")
        if kw.get("progress"):
            kw["progress"]("scenes", {"count": 4})
        return IngestResult("v1", 4, "url", "client", 2.5, spoken_scenes=3)

    def search(self, q):
        return RecordingQuery(self, q)


class RecordingQuery:
    """Stands in for cinematlas.query.Search: records the builder chain, runs on .run()."""

    def __init__(self, eng, q, chain=()):
        self.eng, self.q, self.chain = eng, q, chain

    def __getattr__(self, name):
        return lambda *a: RecordingQuery(self.eng, self.q, (*self.chain, (name, *a)))

    def run(self):
        self.eng.calls.append(("search", self.q, self.chain))
        if ("only", "visual") in self.chain:
            return SearchResults([{**HIT, "scene_id": 2}, {**HIT, "scene_id": 0}])
        return SearchResults([HIT])


@pytest.fixture
def run(monkeypatch, capsys):
    RecordingEngine.instances.clear()
    monkeypatch.setattr(cli, "Cinematlas", RecordingEngine)

    def _run(*argv):
        code = cli.main(list(argv))
        out = capsys.readouterr()
        return code, out.out, out.err, RecordingEngine.instances[-1] if RecordingEngine.instances else None

    return _run


# ---------------------------------------------------------------- doctor / setup
def test_doctor_prints_report_and_exits_0_when_healthy(run):
    code, out, _, eng = run("doctor", "--no-voyage")
    assert code == 0 and "✓ MongoDB" in out and "All good." in out
    assert ("doctor", {"check_voyage": False}) in eng.calls


def test_doctor_exits_1_on_failures_and_supports_json(run, monkeypatch):
    bad = Diagnosis([Check("ffmpeg", "fail", "not on PATH", "brew install ffmpeg")])
    monkeypatch.setattr(RecordingEngine, "diagnosis", bad)
    code, out, _, _ = run("doctor", "--json")
    assert code == 1
    assert json.loads(out) == {"ok": False, "checks": [
        {"name": "ffmpeg", "status": "fail", "detail": "not on PATH", "fix": "brew install ffmpeg"}]}


def test_setup_reports_mode_and_index_states(run):
    code, out, _, eng = run("--db", "d", "--collection", "c", "setup")
    assert code == 0
    assert json.loads(out) == {"transcript_mode": "client", "indexes": {"cinematlas_vector_index": "ready"}}
    assert eng.kwargs["db_name"] == "d" and ("ensure_indexes", {"update": False}) in eng.calls


def test_setup_update_flag_updates_indexes_in_place(run):
    _, _, _, eng = run("setup", "--update")
    assert ("ensure_indexes", {"update": True}) in eng.calls


# ---------------------------------------------------------------- ingest
def test_ingest_prints_result_json_on_stdout_and_progress_on_stderr(run):
    code, out, err, eng = run("ingest", "www.b.com/v.mp4", "--video-id", "v1")
    assert code == 0
    assert json.loads(out)["scenes"] == 4 and json.loads(out)["video_id"] == "v1"
    assert "✓ detected scenes  (count=4)" in err and "Indexed 4 scenes" in err
    assert eng.calls[0][:2] == ("ingest", "www.b.com/v.mp4")  # engine resolves URL vs file


def test_quiet_suppresses_progress(run):
    _, out, err, eng = run("ingest", "https://b.com/v.mp4", "-q")
    assert err == "" and eng.calls[0][2]["progress"] is None
    assert json.loads(out)["scenes"] == 4


def test_dash_streams_stdin_as_upload(run, monkeypatch):
    stdin = SimpleNamespace(buffer=io.BytesIO(b"video-bytes"))
    monkeypatch.setattr(sys, "stdin", stdin)
    _, _, _, eng = run("ingest", "-", "-q")
    assert eng.calls[0][1] is stdin.buffer and eng.calls[0][2]["filename"] == "stdin.mp4"


def test_domain_errors_exit_1_with_message_not_traceback(run):
    code, out, err, eng = run("ingest", "x.mp4", "--video-id", "bad", "-q")
    assert code == 1 and out == ""
    assert "Not a decodable video" in err and "Traceback" not in err
    assert ("close",) in eng.calls


# ---------------------------------------------------------------- search
def test_search_defaults_to_hybrid_and_json_when_piped(run):
    code, out, _, eng = run("search", "how loud")  # capsys stdout is not a TTY
    assert code == 0 and eng.calls[0][0] == "search"
    assert json.loads(out)["moment_link"] == "https://b.com/v.mp4#t=34"


def test_table_format_is_human_readable(run):
    _, out, _, _ = run("search", "how loud", "--format", "table")
    assert out.splitlines()[0].strip() == "1. v#3 @    0:34  110 decibels"
    assert "https://b.com/v.mp4#t=34" in out


def test_context_format_is_pipeable_into_any_llm(run):
    _, out, _, _ = run("search", "how loud", "--format", "context")
    assert out == "[1] v @ 0:34 https://b.com/v.mp4#t=34\n110 decibels\n"


def test_search_routes_by_modality(run):
    _, out, _, eng = run("search", "sonic boom", "--by", "visual", "-k", "2", "--video-id", "v", "--format", "json")
    assert [json.loads(line)["scene_id"] for line in out.splitlines()] == [2, 0]
    assert eng.calls[0] == ("search", "sonic boom", (("limit", 2), ("video", "v"), ("only", "visual")))


def test_search_by_adaptive_opts_into_routing(run):
    _, _, _, eng = run("search", "sonic boom", "--by", "adaptive", "--format", "json")
    assert eng.calls[0] == ("search", "sonic boom", (("limit", 5), ("video", ""), ("adaptive",)))


def test_console_script_is_installed():
    out = subprocess.run([sys.executable, "-m", "cinematlas.cli", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and all(cmd in out.stdout for cmd in ("doctor", "setup", "ingest", "search"))


def test_version_flag():
    import cinematlas

    out = subprocess.run([sys.executable, "-m", "cinematlas.cli", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == f"cinematlas {cinematlas.__version__}"


def test_progress_is_a_tqdm_bar_on_a_terminal_and_lines_otherwise():
    import io

    from cinematlas.cli import _progress_printer

    class Tty(io.StringIO):
        def isatty(self):
            return True

    tty, pipe = Tty(), io.StringIO()
    for stream in (tty, pipe):
        report = _progress_printer(stream)
        for stage in ("fetched", "scenes", "transcribed", "embedded", "stored"):
            report(stage, {"count": 1})
    assert "5/5" in tty.getvalue() and "stored in Atlas" in tty.getvalue()
    assert pipe.getvalue().count("✓") == 5 and "5/5" not in pipe.getvalue()
