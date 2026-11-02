"""Directory output targets are refused before the paid calls (PXG-33, B-058).

In-process ``CliRunner`` only, FakeLLM through the ``cli.make_llm`` seam, offline.
"""

import pytest
from fakes import FakeLLM
from typer.testing import CliRunner

import pixelgen.cli as cli
import pixelgen.pipeline as pipeline
from pixelgen.cli import app

runner = CliRunner()
SLUG = "a-red-apple"


def _grid(seed: int = 0, side: int = 16, colours: int = 16) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5 + seed) % colours
            for c in range(side)
        ]
        for r in range(side)
    ]


class Harness:
    def __init__(self, monkeypatch, tmp_path):
        self.mp = monkeypatch
        self.tmp = tmp_path
        self.llm: FakeLLM | None = None
        self.make_llm_calls = 0
        monkeypatch.chdir(tmp_path)
        self.home = tmp_path / "pxhome"
        monkeypatch.setenv("PIXELGEN_HOME", str(self.home))

    @property
    def out(self):
        return self.tmp / "out"

    def invoke(self, args, *script, key=True):
        self.llm = FakeLLM(*script)
        self.make_llm_calls = 0
        if key:
            self.mp.setenv("ANTHROPIC_API_KEY", "dummy-key")
        else:
            self.mp.delenv("ANTHROPIC_API_KEY", raising=False)

        def fake_make():
            self.make_llm_calls += 1
            return self.llm

        self.mp.setattr(cli, "make_llm", fake_make)
        return runner.invoke(app, args)

    def generate(self, extra=(), *script, key=True):
        script = script or ({}, {"grid": _grid(1)})
        return self.invoke(["generate", "a red apple", "--size", "16", *extra], *script, key=key)

    def refine(self, extra=()):
        return self.invoke(
            ["refine", "x", "--from", str(self.out / f"{SLUG}.json"), *extra],
            {"spec": {}, "grid": _grid(7)},
        )


@pytest.fixture
def h(monkeypatch, tmp_path):
    return Harness(monkeypatch, tmp_path)


def _error_lines(result):
    return [line for line in result.stderr.splitlines() if line.strip()]


@pytest.mark.parametrize(
    ("arm", "target"),
    [
        ("aseprite", "x.aseprite"),
        ("png", "x.png"),
        ("preview", "x@8x.png"),
        ("sidecar", "x.json"),
    ],
)
def test_generate_refuses_directory_target(h, arm, target):
    h.out.mkdir()
    (h.out / target).mkdir()
    extra = ["-o", "out/x"] + (["--aseprite"] if arm == "aseprite" else [])
    r = h.generate(extra)
    assert r.exit_code == 1
    lines = _error_lines(r)
    assert len(lines) == 1
    assert lines[0].startswith("pixelgen: error:")
    assert "cannot write output" in lines[0]
    assert f"{target}'" in lines[0]
    assert h.make_llm_calls == 0
    assert h.llm.call_count == 0
    assert sorted(p.name for p in h.out.iterdir()) == [target]
    assert not list((h.out / target).iterdir())
    assert not (h.home / "last.json").exists()


def test_generate_directory_target_beats_missing_key(h):
    h.out.mkdir()
    (h.out / "x.aseprite").mkdir()
    r = h.generate(["-o", "out/x", "--aseprite"], key=False)
    assert r.exit_code == 1
    lines = _error_lines(r)
    assert len(lines) == 1
    assert lines[0].startswith("pixelgen: error:")
    assert "cannot write output" in lines[0]
    assert "x.aseprite'" in lines[0]
    assert "ANTHROPIC_API_KEY" not in r.stderr
    assert h.make_llm_calls == 0


@pytest.mark.parametrize("flag", ["flag-on", "flag-off"])
def test_refine_refuses_directory_target(h, flag):
    assert h.generate().exit_code == 0
    v2 = h.out / f"{SLUG}-v2"
    h.mp.setattr(pipeline, "next_version_base", lambda _p: v2)
    (h.out / f"{SLUG}-v2.aseprite").mkdir()
    if flag == "flag-on":
        r = h.refine(["--aseprite"])
        assert r.exit_code == 1
        lines = _error_lines(r)
        assert len(lines) == 1
        assert lines[0].startswith("pixelgen: error:")
        assert "cannot write output" in lines[0]
        assert f"{SLUG}-v2.aseprite'" in lines[0]
        assert h.make_llm_calls == 0
        assert h.llm.call_count == 0
        for name in (f"{SLUG}-v2.png", f"{SLUG}-v2@8x.png", f"{SLUG}-v2.json"):
            assert not (h.out / name).exists()
    else:
        r = h.refine()
        assert r.exit_code == 0
        for name in (f"{SLUG}-v2.png", f"{SLUG}-v2@8x.png", f"{SLUG}-v2.json"):
            assert (h.out / name).is_file()
        assert (h.out / f"{SLUG}-v2.aseprite").is_dir()
        assert not list((h.out / f"{SLUG}-v2.aseprite").iterdir())


def test_generate_symlink_to_directory_target_is_replaced(h):
    h.out.mkdir()
    real = h.tmp / "real"
    real.mkdir()
    (h.out / "x.aseprite").symlink_to(real, target_is_directory=True)
    r = h.generate(["-o", "out/x", "--aseprite"])
    assert r.exit_code == 0, r.stderr
    assert h.llm.call_count == 2
    target = h.out / "x.aseprite"
    assert target.is_file()
    assert not target.is_symlink()
    assert list(real.iterdir()) == []


@pytest.mark.parametrize("verb", ["generate", "refine"])
def test_default_and_version_names_step_past_directories(h, verb):
    h.out.mkdir()
    if verb == "generate":
        (h.out / f"{SLUG}.aseprite").mkdir()
        r = h.generate(["--aseprite"])
        assert r.exit_code == 0, r.stderr
        for suffix in (".png", "@8x.png", ".json", ".aseprite"):
            assert (h.out / f"{SLUG}_2{suffix}").is_file()
        assert (h.out / f"{SLUG}.aseprite").is_dir()
        assert not list((h.out / f"{SLUG}.aseprite").iterdir())
        return
    assert h.generate().exit_code == 0
    assert (h.out / f"{SLUG}.json").is_file()
    v2 = [f"{SLUG}-v2{s}" for s in (".png", "@8x.png", ".json", ".aseprite")]
    for name in v2:
        (h.out / name).mkdir()
    r = h.refine(["--aseprite"])
    assert r.exit_code == 0, r.stderr
    assert h.llm.call_count == 1
    for suffix in (".png", "@8x.png", ".json", ".aseprite"):
        assert (h.out / f"{SLUG}-v3{suffix}").is_file()
    for name in v2:
        assert (h.out / name).is_dir()
        assert not list((h.out / name).iterdir())
