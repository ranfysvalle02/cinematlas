"""Command-line entrypoint: ``cinematlas doctor | setup | ingest | search``."""

import argparse
import dataclasses
import json
import logging
import sys
from collections.abc import Sequence
from typing import Any

from . import __version__
from .engine import DEFAULT_WHISPER_MODEL, Cinematlas
from .exceptions import CinematlasError

_STAGE_LABELS = {
    "fetched": "fetched video",
    "scenes": "detected scenes",
    "transcribed": "transcribed speech",
    "embedded": "embedded keyframes + speech",
    "stored": "stored in Atlas",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cinematlas", description="Search inside video, down to the second.")
    p.add_argument("--uri", help="MongoDB URI (default: $MONGODB_URI or $MDB_URI)")
    p.add_argument("--db", default="cinematlas_enterprise")
    p.add_argument("--collection", default="multimodal_scenes")
    p.add_argument("--transcript-mode", choices=["auto", "autoembed", "client"], default="auto")
    p.add_argument("--filter", action="append", default=[], metavar="FIELD",
                   help="a metadata field searches can filter on (repeatable); after adding one, run `setup --update`")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    p.add_argument("--version", action="version", version=f"cinematlas {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    doc = sub.add_parser("doctor", help="Check the deployment and print exact fixes for anything wrong")
    doc.add_argument("--json", action="store_true", help="machine-readable output")
    doc.add_argument("--no-voyage", action="store_true", help="skip the Voyage API key check")

    setup = sub.add_parser("setup", help="Create search indexes (idempotent)")
    setup.add_argument("--update", action="store_true",
                       help="also update outdated indexes in place (e.g. add quantization)")

    ing = sub.add_parser("ingest", help="Ingest a URL (YouTube, direct link, s3://, gs://), a local file, "
                                        "or '-' for stdin")
    ing.add_argument("source")
    ing.add_argument("--video-id")
    ing.add_argument("--filename", help="Display name for file/stdin uploads")
    ing.add_argument("--scene-threshold", type=float, default=27.0)
    ing.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    ing.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                     help="metadata stored on every scene (repeatable), e.g. --meta course=cs101")
    ing.add_argument("-q", "--quiet", action="store_true", help="no progress output")

    se = sub.add_parser("search", help="Search what was shown and said")
    se.add_argument("query")
    se.add_argument("--by", choices=["hybrid", "adaptive", "scene", "transcript", "visual", "text"], default="hybrid",
                    help="hybrid (default): joint image+speech vector ranks scenes, reranker picks the second; "
                         "adaptive: fuse every source, routed per question (leans ahead on speech questions); "
                         "scene / transcript / visual / text: one source alone")
    se.add_argument("--format", choices=["auto", "table", "json", "context"], default="auto",
                    help="auto: table in a terminal, JSON lines when piped; context: citable text for an LLM")
    se.add_argument("-k", "--top-k", type=int, default=5)
    se.add_argument("--video-id")
    se.add_argument("--where", action="append", default=[], metavar="KEY=VALUE",
                    help="only scenes whose metadata matches (repeatable); VALUE may be a,b,c for any of them")

    demo = sub.add_parser("demo", help='Local web app: add a video, ask a question, jump to the second '
                                       '(pip install "cinematlas[demo]")')
    demo.add_argument("--host", default="127.0.0.1")
    demo.add_argument("--port", type=int, default=8765)
    demo.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    return p


def _progress_printer(stream: Any) -> Any:
    """A tqdm bar over the five ingest stages on a terminal (if tqdm is installed), else one line per stage."""
    if getattr(stream, "isatty", lambda: False)():
        try:
            from tqdm import tqdm
        except ImportError:
            pass
        else:
            bar = tqdm(total=len(_STAGE_LABELS), file=stream, unit="stage", leave=True,
                       bar_format="  {bar:20} {n}/{total} {desc}")

            def advance(stage: str, info: dict[str, Any]) -> None:
                detail = ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("source",))
                bar.set_description_str(_STAGE_LABELS.get(stage, stage) + (f" ({detail})" if detail else ""))
                bar.update(1)
                if stage == "stored":
                    bar.close()
            return advance

    def report(stage: str, info: dict[str, Any]) -> None:
        detail = ", ".join(f"{k}={v}" for k, v in info.items() if k not in ("source",))
        print(f"  ✓ {_STAGE_LABELS.get(stage, stage)}" + (f"  ({detail})" if detail else ""), file=stream)
    return report


def _pairs(items: Sequence[str], flag: str) -> dict[str, str]:
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"cinematlas: error: {flag} takes KEY=VALUE, got {item!r}")
        out[key.strip()] = value.strip()
    return out


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    meta = _pairs(getattr(args, "meta", []), "--meta")
    where = {k: v.split(",") if "," in v else v for k, v in _pairs(getattr(args, "where", []), "--where").items()}
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="cinematlas: %(levelname)s: %(message)s" if not args.verbose
        else "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    )
    if not args.verbose:
        logging.getLogger("pyscenedetect").setLevel(logging.WARNING)  # it configures itself at INFO

    try:
        engine = Cinematlas(
            args.uri,
            db_name=args.db,
            collection_name=args.collection,
            transcript_mode=args.transcript_mode,
            whisper_model=getattr(args, "whisper_model", DEFAULT_WHISPER_MODEL),
            filters=tuple(dict.fromkeys([*args.filter, *where])),  # a --where field is a filter field
        )
        with engine:
            if args.command == "doctor":
                report = engine.doctor(check_voyage=not args.no_voyage)
                print(json.dumps(report.to_dict(), indent=2) if args.json else report)
                return 0 if report.ok else 1

            if args.command == "demo":
                from .demo import serve

                engine.ensure_indexes()  # idempotent; a fresh collection needs its indexes before search
                serve(engine, host=args.host, port=args.port, open_browser=not args.no_browser)
                return 0

            if args.command == "setup":
                mode = engine.ensure_indexes(update=args.update)
                print(json.dumps({"transcript_mode": mode,
                                  "indexes": {s.name: s.state for s in engine.inspect_indexes()}}))

            elif args.command == "ingest":
                engine.ensure_indexes()  # idempotent; so `ingest` then `search` works on a fresh cluster
                progress = None if args.quiet else _progress_printer(sys.stderr)
                if not args.quiet:
                    print(f"Ingesting {args.source} …", file=sys.stderr)
                source = sys.stdin.buffer if args.source == "-" else args.source
                filename = args.filename or ("stdin.mp4" if args.source == "-" else None)
                result = engine.ingest(source, video_id=args.video_id, filename=filename, metadata=meta or None,
                                       scene_threshold=args.scene_threshold, progress=progress)
                if not args.quiet:
                    print(f"  {result}", file=sys.stderr)
                print(json.dumps(dataclasses.asdict(result)))

            elif args.command == "search":
                query = engine.search(args.query).limit(args.top_k).video(args.video_id or "")
                if where:
                    query = query.where(**where)
                if args.by == "adaptive":
                    query = query.adaptive()
                elif args.by != "hybrid":
                    query = query.only(args.by)
                hits = query.run()
                fmt = args.format
                if fmt == "auto":
                    fmt = "table" if sys.stdout.isatty() else "json"
                if fmt == "context":
                    print(hits.to_context())
                elif fmt == "table":
                    print(hits)
                else:
                    for hit in hits:
                        print(json.dumps(hit, default=str))
    except (CinematlasError, ValueError) as e:
        print(f"cinematlas: error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
