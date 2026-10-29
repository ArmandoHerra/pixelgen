from pathlib import Path

from pixelgen.config import pixelgen_home


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "custom"))
    p = pixelgen_home()
    assert p == tmp_path / "custom"
    assert not p.exists()


def test_default_under_home(monkeypatch, tmp_path):
    monkeypatch.delenv("PIXELGEN_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    p = pixelgen_home()
    assert p == tmp_path / "h" / ".pixelgen"
    assert not p.exists()


def test_empty_value_uses_default(monkeypatch, tmp_path):
    monkeypatch.setenv("PIXELGEN_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    p = pixelgen_home()
    assert p == tmp_path / "h" / ".pixelgen"
    assert not p.exists()


def test_tilde_is_expanded(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path / "h"))
    monkeypatch.setenv("PIXELGEN_HOME", "~/x")
    p = pixelgen_home()
    assert p == Path(tmp_path / "h" / "x")
    assert not p.exists()
