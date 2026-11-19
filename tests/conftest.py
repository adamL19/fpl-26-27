import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.database import connect                      # noqa: E402
from etl.pipeline import refresh                      # noqa: E402
from model.analysis import load_context               # noqa: E402
from model.config import ModelParams                  # noqa: E402
from tests.fake_api import FakeFPL                    # noqa: E402


@pytest.fixture(scope="session")
def fake():
    return FakeFPL()


@pytest.fixture(scope="session")
def conn(fake, tmp_path_factory):
    c = connect(tmp_path_factory.mktemp("db") / "t.sqlite")
    refresh(fake, c, progress=lambda *_: None)
    return c


@pytest.fixture(scope="session")
def ctx(conn):
    return load_context(conn, ModelParams(horizon=6))
