"""Remote URLs are untrusted input: normalize them and refuse what's unsafe to fetch (SSRF guard)."""

from __future__ import annotations

import ipaddress
import os
import socket
import urllib.parse

from ._utils import normalize_source_url
from .exceptions import IngestionError

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".ts", ".flv", ".wmv"}


def validate_remote_url(source: str, *, allow_private: bool = False) -> str:
    """Normalize a remote source and refuse schemes/hosts that are unsafe to fetch.

    Private, loopback, link-local and reserved addresses are refused unless ``allow_private``: when
    user-supplied URLs reach ingestion, this blocks SSRF against internal services and cloud metadata
    endpoints (e.g. 169.254.169.254). Direct video links re-check every redirect with this function;
    redirects followed by yt-dlp (pages such as YouTube) are not, so put an egress proxy in front if you
    accept arbitrary page URLs from the public. ``s3://`` and ``gs://`` are fetched by the cloud SDK with
    your own credentials, so there is no host to check.
    """
    url = normalize_source_url(source)
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme in ("s3", "gs"):  # fetched by the cloud SDK with your credentials: no host to SSRF-check
        if not parsed.netloc or not parsed.path.strip("/"):
            raise IngestionError(f"Expected {parsed.scheme}://bucket/key, got {source!r}")
        return url
    if "://" not in url:
        raise IngestionError(f"No such file or URL: {source!r}")
    if os.path.splitext(parsed.hostname or "")[1].lower() in VIDEO_EXTENSIONS and "/" not in source:
        raise IngestionError(f"No such file: {source!r}")  # "clip.mp4" is a missing file, not a host
    if parsed.scheme not in ("http", "https"):
        raise IngestionError(f"Unsupported URL scheme {parsed.scheme!r}; use http(s), s3://, gs:// "
                             "or ingest_file()")
    if not parsed.hostname:
        raise IngestionError(f"URL has no host: {source!r}")
    if not allow_private:
        try:
            infos = socket.getaddrinfo(parsed.hostname, parsed.port or None, proto=socket.IPPROTO_TCP)
        except socket.gaierror as e:
            raise IngestionError(f"Cannot resolve host {parsed.hostname!r}: {e}") from e
        for info in infos:
            ip = ipaddress.ip_address(info[4][0])
            if not ip.is_global:
                raise IngestionError(
                    f"Refusing to fetch {parsed.hostname!r} ({ip}): non-public address. "
                    "Pass allow_private_urls=True for trusted internal sources."
                )
    return url
