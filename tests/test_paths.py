"""core.paths — platformdirs 경로/환경변수 오버라이드 (P10)."""
import importlib

from pystitch.core import paths


def test_env_override_config(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.CONFIG_DIR_ENV_VAR, str(tmp_path / "cfg"))
    assert paths.get_config_dir() == tmp_path / "cfg"
    assert paths.settings_ini_path() == tmp_path / "cfg" / "settings.ini"


def test_env_override_data(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.DATA_DIR_ENV_VAR, str(tmp_path / "data"))
    assert paths.get_data_dir() == tmp_path / "data"


def test_default_paths_have_brand(monkeypatch):
    monkeypatch.delenv(paths.CONFIG_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(paths.DATA_DIR_ENV_VAR, raising=False)
    importlib.reload(paths)
    c = str(paths.get_config_dir())
    assert "TouchlineLabs" in c and "TouchlineAnalyst" in c
    d = str(paths.get_data_dir())
    assert d.endswith("TouchlineLabs/TouchlineAnalyst") \
        or d.endswith("TouchlineLabs\\TouchlineAnalyst")


def test_version_importable():
    from version import __version__, __version_info__
    assert isinstance(__version_info__, tuple)
    assert __version__.count(".") >= 2
