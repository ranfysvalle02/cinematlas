"""Backends for integration tests.

* ``atlas_cloud`` — the real Atlas cluster from ``.env`` (MDB_URI / MONGODB_URI).
  Atlas autoEmbed only exists here, so this backend exercises ``autoembed`` mode.
* ``atlas_local`` — ``mongodb/mongodb-atlas-local`` in Docker. It has Vector Search
  but rejects autoEmbed, so this backend exercises the *real* fallback to
  ``client`` mode (client-side voyage-4 transcript vectors).

Both use a stable collection (search indexes are slow to build) and isolate each
test with a unique ``video_id`` that is deleted afterwards.
"""

import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError
from support import env_or_none

from cinematlas import Cinematlas

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover
    pass

DB, COLL = "cinematlas_ci", "scenes"
LOCAL_IMAGE = "mongodb/mongodb-atlas-local:latest"
LOCAL_CONTAINER = "cinematlas-test-atlas-local"
LOCAL_PORT = int(os.getenv("CINEMATLAS_ATLAS_LOCAL_PORT", "27028"))
# Atlas Local's search process ("mongot") starts after mongod; index commands fail with this until then.
SEARCH_NOT_READY = 508


def _require_voyage():
    if not os.getenv("VOYAGE_API_KEY"):
        pytest.skip("VOYAGE_API_KEY not configured")


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.fixture(scope="session")
def atlas_local_uri():
    """Start (or reuse) an Atlas Local container; tear it down only if we started it."""
    if os.getenv("ATLAS_LOCAL_URI"):
        yield os.environ["ATLAS_LOCAL_URI"]
        return
    if not _docker_available():
        pytest.skip("Docker not available for mongodb-atlas-local")

    running = subprocess.run(
        ["docker", "ps", "-q", "-f", f"name=^{LOCAL_CONTAINER}$"], capture_output=True, text=True
    ).stdout.strip()
    started = False
    if not running:
        subprocess.run(["docker", "rm", "-f", LOCAL_CONTAINER], capture_output=True)
        subprocess.run(
            ["docker", "run", "-d", "--name", LOCAL_CONTAINER, "-p", f"{LOCAL_PORT}:27017", LOCAL_IMAGE],
            check=True, capture_output=True,
        )
        started = True

    uri = f"mongodb://localhost:{LOCAL_PORT}/?directConnection=true"
    client = MongoClient(uri, serverSelectionTimeoutMS=2000)
    deadline = time.monotonic() + 120
    while True:
        try:
            client.admin.command("ping")
            break
        except PyMongoError:
            if time.monotonic() > deadline:
                raise
            time.sleep(2)
    client.close()

    yield uri
    if started:
        subprocess.run(["docker", "rm", "-f", LOCAL_CONTAINER], capture_output=True)


def _engine_with_indexes(uri: str, **kwargs) -> tuple[Cinematlas, str]:
    eng = Cinematlas(uri, db_name=DB, collection_name=COLL, **kwargs)
    deadline = time.monotonic() + 180
    while True:
        try:
            return eng, eng.ensure_indexes(timeout_s=900)
        except OperationFailure as e:
            if e.code != SEARCH_NOT_READY or time.monotonic() > deadline:
                raise
            time.sleep(3)


@pytest.fixture(scope="session")
def atlas_cloud():
    _require_voyage()
    uri = env_or_none("MONGODB_URI", "MDB_URI")
    if not uri:
        pytest.skip("MDB_URI / MONGODB_URI not configured")
    eng, mode = _engine_with_indexes(uri, text_model=os.getenv("VOYAGE_MODEL", "voyage-4"))
    yield eng, mode
    eng.close()


@pytest.fixture(scope="session")
def atlas_local(atlas_local_uri):
    _require_voyage()
    eng, mode = _engine_with_indexes(atlas_local_uri, text_model=os.getenv("VOYAGE_MODEL", "voyage-4"))
    yield eng, mode
    eng.close()


@pytest.fixture(
    params=[
        pytest.param("atlas_cloud", marks=pytest.mark.integration),
        pytest.param("atlas_local", marks=[pytest.mark.integration, pytest.mark.atlas_local]),
    ]
)
def backend(request):
    """(engine, expected_mode) for each deployment type."""
    return request.getfixturevalue(request.param)


@pytest.fixture
def run_id():
    return f"it-{uuid.uuid4().hex[:10]}"


@pytest.fixture
def cleanup(run_id):
    engines = []
    yield engines.append
    for eng in engines:
        eng.collection.delete_many({"video_id": {"$regex": f"^{run_id}"}})
