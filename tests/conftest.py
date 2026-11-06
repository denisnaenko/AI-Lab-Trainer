import pytest

from labgen.config import load_settings
from labgen.service import LabService
from labgen.storage import LabStore


@pytest.fixture
def settings(tmp_path):
    s = load_settings(env_file=None)
    s.data_dir = tmp_path / "data"
    s.llm_base_url = ""
    s.llm_model = ""
    return s


@pytest.fixture
def service(settings):
    return LabService(settings, store=LabStore(settings.data_dir), use_settings_provider=False)
