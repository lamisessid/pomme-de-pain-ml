import os
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))


@pytest.fixture(scope="session")
def client(tmp_path_factory):
    os.environ["DB_PATH"] = str(tmp_path_factory.mktemp("db") / "test.db")
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import app

    settings.DB_PATH = Path(os.environ["DB_PATH"])
    with TestClient(app) as c:
        yield c
