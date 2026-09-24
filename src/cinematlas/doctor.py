"""Diagnostics: what this deployment supports, what's misconfigured, and how to fix it."""

import shutil
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["ok", "warn", "fail", "info"]
_ICON = {"ok": "✓", "warn": "!", "fail": "✗", "info": "·"}

# Minimum server versions for Atlas-native stages.
NATIVE_STAGE_VERSION = {"$rankFusion": "8.0", "$rerank": "8.3"}


@dataclass(frozen=True)
class Check:
    name: str
    status: Status
    detail: str
    fix: str | None = None


@dataclass
class Diagnosis:
    """Result of :meth:`Cinematlas.doctor`. Print it, or inspect ``checks``."""

    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No failures (warnings are allowed: everything degrades gracefully)."""
        return not any(c.status == "fail" for c in self.checks)

    @property
    def problems(self) -> list[Check]:
        return [c for c in self.checks if c.status in ("warn", "fail")]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "checks": [asdict(c) for c in self.checks]}

    def __str__(self) -> str:
        width = max((len(c.name) for c in self.checks), default=0)
        lines = []
        for c in self.checks:
            lines.append(f" {_ICON[c.status]} {c.name.ljust(width)}  {c.detail}")
            if c.fix and c.status in ("warn", "fail"):
                lines.append(f"   {' ' * width}  → {c.fix}")
        n_warn = sum(c.status == "warn" for c in self.checks)
        n_fail = sum(c.status == "fail" for c in self.checks)
        summary = ("All good." if not (n_warn or n_fail)
                   else f"{n_fail} problem(s), {n_warn} warning(s). Cinematlas degrades gracefully on warnings.")
        return "\n".join(lines + ["", summary])


def _version_tuple(version: str | None) -> tuple[int, ...]:
    try:
        return tuple(int(p) for p in (version or "").split(".")[:2])
    except ValueError:
        return ()


def explain_native_failure(stage: str, error: Exception, server_version: str | None = None) -> tuple[str, str]:
    """Turn a server error for ``$rankFusion`` / ``$rerank`` into (reason, fix).

    With ``server_version``, an "unrecognized stage" on a new-enough server is attributed to
    the deployment type (e.g. ``$rerank`` exists only on Atlas), not to the version.
    """
    msg = str(error)
    low = msg.lower()
    version = NATIVE_STAGE_VERSION.get(stage, "")
    new_enough = bool(server_version) and _version_tuple(server_version) >= _version_tuple(version)
    if new_enough and ("unrecognized pipeline stage" in low or "unknown pipeline stage" in low):
        return (f"{stage} isn't available on this deployment (MongoDB {server_version}; Atlas-only feature)",
                "Expected on Atlas Local / self-managed: the fallback gives identical results. "
                "On Atlas, check the cluster tier and project settings.")
    if "not enabled" in low or "project setting" in low:
        return (f"{stage} is disabled for this Atlas project",
                "Atlas UI → Project Settings → enable Native Reranking. Nothing else to change.")
    if "unrecognized pipeline stage" in low or "unknown pipeline stage" in low:
        return (f"{stage} needs MongoDB {version}+",
                "Upgrade the cluster (Atlas: pick 'Latest' with auto-upgrades), or keep the client-side "
                "fallback.")
    if "self-managed" in low or "atlas local" in low or "not supported" in low:
        return (f"{stage} isn't available on this deployment type (e.g. Atlas Local)",
                "Expected outside Atlas: the client-side fallback gives identical results.")
    return (f"{stage} failed: {msg.split(', full error')[0][:160]}",
            "Check cluster version and project settings; the client-side fallback is active meanwhile.")


def run_doctor(engine: Any, *, check_voyage: bool = True) -> Diagnosis:
    """Check everything a :class:`~cinematlas.Cinematlas` depends on; see :meth:`Cinematlas.doctor`."""
    from pymongo.errors import PyMongoError

    from .capabilities import ROUTING_CALIBRATION
    from .exceptions import IngestionStatus

    checks: list[Check] = []
    add = checks.append

    # MongoDB
    try:
        info = engine.mongo_client.admin.command("buildInfo")
        version = info.get("version", "?")
        engine._server_version = str(version)
        add(Check("MongoDB", "ok", f"connected · server {version} · {engine.db_name}.{engine.collection_name}"))
    except PyMongoError as e:
        add(Check("MongoDB", "fail", f"cannot reach the cluster: {str(e)[:120]}",
                  "Check MONGODB_URI / MDB_URI, network access list, and credentials."))
        return Diagnosis(checks)

    # Transcript backend
    mode = engine.transcript_mode
    add(Check("Transcript search", "ok" if mode == "autoembed" else "info",
              "Atlas autoEmbed (voyage-4, embedded server-side)" if mode == "autoembed"
              else "client-side voyage-4 vectors (works on any deployment)"))

    # Indexes
    try:
        statuses = engine.inspect_indexes()
    except PyMongoError as e:
        statuses = []
        add(Check("Indexes", "fail", f"cannot list search indexes: {str(e)[:120]}",
                  "Search indexes need Atlas or Atlas Local (mongot)."))
    for ix in statuses:
        if ix.state == "missing":
            add(Check(f"Index {ix.name}", "fail", "missing", "engine.ensure_indexes()  ·  cinematlas setup"))
        elif ix.state == "failed":
            add(Check(f"Index {ix.name}", "fail", "build failed", "Inspect it in the Atlas UI, then recreate it."))
        elif ix.drift:
            add(Check(f"Index {ix.name}", "warn", "outdated: " + "; ".join(ix.drift),
                      "engine.ensure_indexes(update=True)  ·  cinematlas setup --update  (in place, no downtime)"))
        elif ix.state == "building":
            add(Check(f"Index {ix.name}", "warn", "building (queries may miss recent data)",
                      "Wait; engine.ensure_indexes() blocks until ready."))
        else:
            add(Check(f"Index {ix.name}", "ok", "ready"))

    # Atlas-native stages
    ok, err = _probe(engine, "$rankFusion", [
        {"$rankFusion": {"input": {"pipelines": {"probe": [{"$sort": {"_id": 1}}, {"$limit": 1}]}}}},
        {"$limit": 1}])
    engine.native_fusion = ok
    if ok:
        add(Check("$rankFusion", "ok", "hybrid search runs as one native query"))
    else:
        reason, fix = explain_native_failure("$rankFusion", err, engine.server_version)
        add(Check("$rankFusion", "warn", f"{reason}; using client-side fusion (same ranking)", fix))

    has_docs = engine.collection.find_one({}, {"_id": 1}) is not None
    if not engine.rerank_model:
        add(Check("$rerank", "info", "reranking disabled (rerank_model=None)"))
    elif not has_docs:
        add(Check("$rerank", "info", "not probed yet (collection is empty)"))
    else:
        ok, err = _probe(engine, "$rerank", [
            {"$limit": 1}, {"$set": {"_probe": "probe"}},
            {"$rerank": {"query": {"text": "probe"}, "path": "_probe", "numDocsToRerank": 1,
                         "model": engine.rerank_model}}])
        engine.native_rerank = ok
        if ok:
            add(Check("$rerank", "ok", f"sentence reranking runs inside Atlas ({engine.rerank_model})"))
        else:
            reason, fix = explain_native_failure("$rerank", err, engine.server_version)
            add(Check("$rerank", "warn", f"{reason}; using the Voyage rerank API (same model)", fix))

    # Routing
    if not engine.rerank_model:
        add(Check("Routing", "info", "off (no reranker): search() uses fixed fusion"))
    elif engine.routing_thresholds is not None:
        lo, hi = engine.routing_thresholds
        add(Check("Routing", "ok", f"scene-first default; adaptive custom calibration ({lo:.2f}–{hi:.2f})"))
    elif engine.rerank_model in ROUTING_CALIBRATION:
        lo, hi = ROUTING_CALIBRATION[engine.rerank_model]
        add(Check("Routing", "ok",
                  f"scene-first default; adaptive calibrated for {engine.rerank_model} ({lo:.2f}–{hi:.2f})"))
    else:
        add(Check("Routing", "warn", f"no calibration for {engine.rerank_model}; using fixed fusion",
                  "Calibrate on your data with bench/ and pass routing_thresholds=(lo, hi), "
                  "or use rerank_model='rerank-2.5'."))

    # Voyage
    if check_voyage:
        try:
            engine.vo.embed(["ping"], model=engine.text_model, input_type="query")
            add(Check("Voyage AI", "ok", f"API key works ({engine.model}, {engine.text_model})"))
        except Exception as e:
            add(Check("Voyage AI", "fail", f"embedding call failed: {str(e)[:120]}",
                      "Set VOYAGE_API_KEY (https://dashboard.voyageai.com)."))

    # Data
    if has_docs:
        done = {"status": IngestionStatus.COMPLETED.value}
        scenes = engine.collection.count_documents(done)
        videos = len(engine.collection.distinct("video_id", done))
        add(Check("Data", "ok", f"{scenes} scenes across {videos} videos"))
        legacy = engine.collection.count_documents({**done, "segments": {"$exists": False}})
        if engine.scene_embeddings:
            legacy = max(legacy, engine.collection.count_documents({**done, "scene_embedding": {"$exists": False}}))
        if legacy:
            add(Check("Legacy scenes", "warn",
                      f"{legacy} scenes predate 0.2 (no moments / joint vectors); still searchable",
                      "Re-ingest those videos: engine.ingest(url_or_file, video_id=...)"))
        failed = engine.collection.distinct("video_id", {"status": IngestionStatus.FAILED.value})
        if failed:
            shown = ", ".join(map(str, failed[:5])) + (" …" if len(failed) > 5 else "")
            add(Check("Failed ingests", "warn", f"{len(failed)} video(s) with FAILED tombstones: {shown}",
                      "See error_message on the tombstone; a successful re-ingest clears it."))
    else:
        add(Check("Data", "info", "no scenes yet", "engine.ingest('https://…/video.mp4')"))

    # Local media tooling
    add(Check("ffmpeg", "ok", shutil.which("ffmpeg")) if shutil.which("ffmpeg") else
        Check("ffmpeg", "fail", "not on PATH (needed to extract audio)",
              "brew install ffmpeg  ·  apt-get install -y ffmpeg"))
    if engine.openai_client:
        add(Check("Speech-to-text", "ok", "OpenAI Whisper API"))
    else:
        try:
            import faster_whisper  # noqa: F401

            add(Check("Speech-to-text", "ok", f"faster-whisper ({engine.whisper_model}), runs locally"))
        except ImportError:
            add(Check("Speech-to-text", "warn", "no backend installed: videos index visually only",
                      'pip install "cinematlas[whisper]"  (or set OPENAI_API_KEY with cinematlas[openai])'))
    return Diagnosis(checks)


def _probe(engine: Any, stage: str, pipeline: list[dict[str, Any]]) -> tuple[bool, Exception | None]:
    from pymongo.errors import PyMongoError

    try:
        list(engine.collection.aggregate(pipeline))
        return True, None
    except PyMongoError as e:
        return False, e
