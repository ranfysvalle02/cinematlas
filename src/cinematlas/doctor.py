"""Diagnostics: what this deployment supports, what's misconfigured, and how to fix it."""

from dataclasses import asdict, dataclass, field
from typing import Literal

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
