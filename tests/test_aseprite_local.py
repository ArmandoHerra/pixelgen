"""PXG-24: the shared local-Aseprite seam, the regen driver's no-Aseprite arms, the workflow pin.

Nothing here needs Aseprite: the stubs are /bin/sh scripts.
"""

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import aseprite_local as local
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
DRIVER = TESTS / "regen_aseprite_made.py"
WORKFLOWS = ROOT / ".github" / "workflows"
SKIP_LITERAL = (
    "SKIP local Aseprite: PIXELGEN_ASEPRITE is unset; Aseprite never runs in CI (PXG-24, OR-20)"
)


VERSION_THEN = '#!/bin/sh\n[ "$1" = "--version" ] && echo "Aseprite 0.0"\n'


def _exe(path: Path, body: str = "#!/bin/sh\nexit 0\n", mode: int = 0o755) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(mode)
    return path


# ------------------------------------------------------------------------- AC20 (a)


def test_local_seam_interface():
    import inspect

    assert local.ASEPRITE_ENV_VAR == "PIXELGEN_ASEPRITE"
    assert local.LOCAL_SKIP_LINE == SKIP_LITERAL
    for name in ("require_aseprite", "aseprite_from_env", "aseprite_env", "run_aseprite"):
        assert callable(getattr(local, name)), name
    params = inspect.signature(local.run_aseprite).parameters
    assert params["root"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["timeout"].default == 120


# ------------------------------------------------------------------------- AC20 (b), (c)


@pytest.mark.parametrize("value", [None, ""], ids=["unset", "empty"])
def test_require_aseprite_skips_when_unset(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("PIXELGEN_ASEPRITE", raising=False)
    else:
        monkeypatch.setenv("PIXELGEN_ASEPRITE", value)
    with pytest.raises(pytest.skip.Exception) as info:
        local.require_aseprite()
    assert info.value.msg == SKIP_LITERAL


def _non_exec_target(kind: str, tmp_path: Path) -> Path:
    plain = tmp_path / "plain"
    plain.write_text("x", encoding="utf-8")
    plain.chmod(0o644)
    if kind == "missing":
        return tmp_path / "nope"
    if kind == "directory":
        return tmp_path
    if kind == "not_exec":
        return plain
    link = tmp_path / "link-to-plain"
    link.symlink_to(plain)
    return link


@pytest.mark.parametrize("kind", ["missing", "directory", "not_exec", "symlink_to_not_exec"])
def test_require_aseprite_fails_not_skips_on_non_executable(monkeypatch, tmp_path, kind):
    monkeypatch.setenv("PIXELGEN_ASEPRITE", str(_non_exec_target(kind, tmp_path)))
    try:
        with pytest.raises(pytest.fail.Exception):
            local.require_aseprite()
    except pytest.skip.Exception:
        pytest.fail("skipped a non-executable path")


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_require_aseprite_returns_executable(monkeypatch, tmp_path, kind):
    real = _exe(tmp_path / "real-aseprite")
    target = real
    if kind == "symlink":
        target = tmp_path / "alias"
        target.symlink_to(real)
    monkeypatch.setenv("PIXELGEN_ASEPRITE", str(target))
    assert local.require_aseprite() == target
    assert str(local.aseprite_from_env()) == str(target)  # not resolved


def test_aseprite_from_env_rejects_non_executable_with_value_error(monkeypatch, tmp_path):
    bad = tmp_path / "bad"
    bad.write_text("x", encoding="utf-8")
    bad.chmod(0o644)
    monkeypatch.setenv("PIXELGEN_ASEPRITE", str(bad))
    with pytest.raises(ValueError, match="is not an executable file"):
        local.aseprite_from_env()
    monkeypatch.delenv("PIXELGEN_ASEPRITE")
    assert local.aseprite_from_env() is None


# ------------------------------------------------------------------------- AC20 (d)


def test_aseprite_env_scrubs_display_and_roots_home_and_xdg(monkeypatch, tmp_path):
    sentinels = {
        "DISPLAY": ":sentinel",
        "WAYLAND_DISPLAY": "wayland-sentinel",
        "HOME": "/sentinel/home",
        "XDG_CONFIG_HOME": "/sentinel/c",
        "XDG_CACHE_HOME": "/sentinel/k",
        "XDG_DATA_HOME": "/sentinel/d",
        "XDG_STATE_HOME": "/sentinel/s",
    }
    for key, value in sentinels.items():
        monkeypatch.setenv(key, value)
    root = tmp_path / "r"
    env = local.aseprite_env(root)
    assert "DISPLAY" not in env and "WAYLAND_DISPLAY" not in env
    expected = {
        "HOME": root / "home",
        "PIXELGEN_HOME": root / "pixelgen-home",
        "XDG_CONFIG_HOME": root / "xdg" / "config",
        "XDG_CACHE_HOME": root / "xdg" / "cache",
        "XDG_DATA_HOME": root / "xdg" / "data",
        "XDG_STATE_HOME": root / "xdg" / "state",
    }
    assert len({env[k] for k in expected}) == 6
    for key, path in expected.items():
        assert env[key] == str(path)
        assert Path(env[key]).is_dir()
        assert Path(env[key]).resolve().is_relative_to(root.resolve())
    assert env["PATH"] == os.environ["PATH"]
    for key, value in sentinels.items():
        assert os.environ[key] == value


# ------------------------------------------------------------------------- AC20 (e)


def test_run_aseprite_child_sees_scrubbed_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DISPLAY", ":99")
    (tmp_path / "real").mkdir()
    (tmp_path / "link").symlink_to(tmp_path / "real")
    root = tmp_path / "link"
    stub = _exe(
        tmp_path / "stub.sh",
        "#!/bin/sh\n"
        'for a in "$@"; do printf "ARG=%s\\n" "$a"; done\n'
        "for k in HOME PIXELGEN_HOME XDG_CONFIG_HOME XDG_CACHE_HOME "
        "XDG_DATA_HOME XDG_STATE_HOME; do\n"
        '  eval "v=\\${$k}"; printf "%s=%s\\n" "$k" "$v"\n'
        "done\n"
        'printf "DISPLAY=%s\\n" "${DISPLAY-unset}"\n'
        'printf "WAYLAND_DISPLAY=%s\\n" "${WAYLAND_DISPLAY-unset}"\n'
        'printf "PWD=%s\\n" "$(pwd -P)"\n',
    )
    result = local.run_aseprite(stub, ["a b", "c"], root=root)
    assert result.returncode == 0, result.stderr
    pairs = [line.split("=", 1) for line in result.stdout.splitlines()]
    assert [v for k, v in pairs if k == "ARG"] == ["a b", "c"]
    got = {k: v for k, v in pairs if k != "ARG"}
    assert got["DISPLAY"] == "unset" and got["WAYLAND_DISPLAY"] == "unset"
    for key in (
        "HOME",
        "PIXELGEN_HOME",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    ):
        assert Path(got[key]).resolve().is_relative_to(root.resolve()), key
    assert Path(got["PWD"]).resolve() == root.resolve()
    assert root.is_symlink() and not got["PWD"].startswith(str(root))


# ------------------------------------------------------------------------- AC20 (f)


def _imported(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            found.add((node.module or "").split(".")[0])
    return found


def test_local_helpers_import_no_product_code():
    stdlib = set(sys.stdlib_module_names)
    seam = _imported(TESTS / "aseprite_local.py")
    driver = _imported(DRIVER)
    assert seam <= stdlib | {"pytest"}, sorted(seam - stdlib)
    assert driver <= stdlib | {"aseprite_local"}, sorted(driver - stdlib)
    assert not (seam | driver) & {"pixelgen", "numpy", "PIL"}


# ------------------------------------------------------------------------- AC22


def _driver_env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "PIXELGEN_ASEPRITE"}
    env["HOME"] = str(tmp_path / "home")
    env["PIXELGEN_HOME"] = str(tmp_path / "pxhome")
    env.update(extra)
    return env


def _run_driver(tmp_path: Path, **extra: str) -> tuple[subprocess.CompletedProcess, Path]:
    out = tmp_path / "o"
    result = subprocess.run(
        [sys.executable, "-I", str(DRIVER), "--out", str(out)],
        env=_driver_env(tmp_path, **extra),
        capture_output=True,
        text=True,
        check=False,
    )
    return result, out


def test_regen_driver_skips_without_aseprite(tmp_path):
    result, out = _run_driver(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == SKIP_LITERAL + "\n"
    assert not out.exists()


def test_regen_driver_rejects_non_executable(tmp_path):
    bad = tmp_path / "not-exec"
    bad.write_text("x", encoding="utf-8")
    bad.chmod(0o644)
    result, out = _run_driver(tmp_path, PIXELGEN_ASEPRITE=str(bad))
    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("regen_aseprite_made: error: ")
    assert "Traceback" not in result.stderr
    assert not out.exists()


def test_regen_driver_reports_a_failing_aseprite_in_one_line(tmp_path):
    stub = _exe(
        tmp_path / "fail.sh", '#!/bin/sh\n[ "$1" = "--version" ] && echo "Aseprite 0.0"\nexit 255\n'
    )
    result, _ = _run_driver(tmp_path, PIXELGEN_ASEPRITE=str(stub))
    assert result.returncode == 1
    assert len(result.stderr.splitlines()) == 1
    assert result.stderr.startswith("regen_aseprite_made: error: ")
    assert "Traceback" not in result.stderr


def test_regen_driver_detects_silent_exit_zero_without_output(tmp_path):
    stub = _exe(
        tmp_path / "silent.sh", '#!/bin/sh\n[ "$1" = "--version" ] && echo "Aseprite 0.0"\nexit 0\n'
    )
    result, _ = _run_driver(tmp_path, PIXELGEN_ASEPRITE=str(stub))
    assert result.returncode == 1
    assert result.stderr.startswith("regen_aseprite_made: error: ")


# ------------------------------------------------------------------------- AC23 (D12)

ASEPRITE_EXE = re.compile(r"(?<![\w.$-])(?:[\w./$-]*/)?aseprite(?![\w.-])")


def _run_texts(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "run" and isinstance(value, str):
                yield value
            else:
                yield from _run_texts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _run_texts(item)


def _violations() -> list[str]:
    found = []
    for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        text = path.read_text(encoding="utf-8")
        if "PIXELGEN_ASEPRITE" in text:
            found.append(f"{path.name}: mentions PIXELGEN_ASEPRITE")
        for run in _run_texts(yaml.safe_load(text)):
            if ASEPRITE_EXE.search(run):
                found.append(f"{path.name}: a step runs an aseprite executable")
    return found


def test_no_workflow_runs_local_aseprite():
    assert any(WORKFLOWS.glob("*.yml"))  # the glob sees the real workflows
    assert _violations() == []


@pytest.mark.parametrize(
    "sample",
    [
        "aseprite_compat.py",
        "x.aseprite",
        '"${f%.aseprite}.libresprite.png"',
        "aseprite-compat",
    ],
)
def test_aseprite_invocation_pattern_samples_no_hit(sample):
    assert ASEPRITE_EXE.search(sample) is None


@pytest.mark.parametrize(
    "sample",
    [
        "aseprite -b x.aseprite --save-as x.png",
        "/usr/local/bin/aseprite --version",
        "sudo aseprite --version",
        "echo $(aseprite --version)",
    ],
)
def test_aseprite_invocation_pattern_samples_hit(sample):
    assert ASEPRITE_EXE.search(sample) is not None


def test_workflow_pin_flags_a_violating_workflow(tmp_path, monkeypatch):
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "bad.yml").write_text(
        "jobs:\n  t:\n    steps:\n      - run: aseprite --version\n", encoding="utf-8"
    )
    (wf / "env.yaml").write_text("env:\n  PIXELGEN_ASEPRITE: /x\n", encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "WORKFLOWS", wf)
    assert sorted(_violations()) == [
        "bad.yml: a step runs an aseprite executable",
        "env.yaml: mentions PIXELGEN_ASEPRITE",
    ]
