"""PXG-28: pins for the advisory LibreSprite workflow and its compare helper (in-process).

Nothing here runs LibreSprite or Aseprite, spawns a child Python, or touches the network.
"""

import ast
import importlib.util
import re
import sys
from pathlib import Path

import aseprite_oracle as ao
import numpy as np
import yaml
from PIL import Image

from pixelgen.pixelate import grid_to_rgba

ROOT = Path(__file__).resolve().parents[1]
GITHUB = ROOT / ".github"
COMPAT = GITHUB / "workflows" / "aseprite-compat.yml"
CI = GITHUB / "workflows" / "ci.yml"
SCRIPT = GITHUB / "scripts" / "aseprite_compat.py"

LS_URL = (
    "https://github.com/LibreSprite/LibreSprite/releases/download/v1.3/"
    "libresprite-v1.3-linux-x86_64.AppImage"
)
LS_SHA256 = "fd90bddda8135fe00f4a9a9c8c9ebd48f9b07e8782ab269be5c9efb1a851af30"


def _load(path: Path = COMPAT) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(wf: dict) -> dict:
    return wf["on"] if "on" in wf else wf[True]


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def _run_text(job: dict) -> str:
    return "\n".join(s["run"] for s in job.get("steps", []) if "run" in s)


def _uses(wf: dict) -> list[str]:
    return [n["uses"] for n in _walk(wf) if isinstance(n, dict) and "uses" in n]


def _load_script():
    spec = importlib.util.spec_from_file_location("aseprite_compat", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compat = _load_script()


def _img(pixels: list[list[tuple[int, int, int, int]]]) -> Image.Image:
    return Image.fromarray(np.array(pixels, dtype=np.uint8))


def test_compat_triggers_are_dispatch_and_weekly_schedule_only():
    trig = _triggers(_load())
    assert set(trig) == {"workflow_dispatch", "schedule"}
    crons = trig["schedule"]
    assert len(crons) == 1
    fields = crons[0]["cron"].split()
    assert len(fields) == 5
    assert fields[4] != "*"


def test_compat_permissions_read_only_and_runner_pinned():
    wf = _load()
    assert wf["permissions"] == {"contents": "read"}
    assert wf["jobs"]
    for name, job in wf["jobs"].items():
        assert "permissions" not in job, name
        assert job["runs-on"] == "ubuntu-24.04", name
        assert "timeout-minutes" in job, name


def test_compat_has_no_secrets_upload_or_allowed_failure():
    text = COMPAT.read_text(encoding="utf-8")
    wf = _load()
    assert "secrets." not in text
    for use in _uses(wf):
        assert "upload-artifact" not in use and "cache" not in use, use
    assert not [n for n in _walk(wf) if isinstance(n, dict) and "continue-on-error" in n]
    assert "|| true" not in "\n".join(_run_text(job) for job in wf["jobs"].values())


def test_compat_actions_pinned_and_already_used_by_ci():
    uses = _uses(_load())
    assert uses
    for use in uses:
        line = next((ln for ln in COMPAT.read_text(encoding="utf-8").splitlines() if use in ln), "")
        assert re.search(r"@[0-9a-f]{40}\s+#\s*v\d", line), line
    assert set(uses) <= set(_uses(_load(CI)))


def test_compat_appimage_sha256_checked_before_execution():
    job = _load()["jobs"]["libresprite"]
    assert job["env"]["LS_URL"] == LS_URL
    assert job["env"]["LS_SHA256"] == LS_SHA256
    text = _run_text(job)
    check = text.index("sha256sum -c")
    assert check < text.index("chmod +x")
    if "--appimage-extract" in text:
        assert check < text.index("--appimage-extract")
    assert check < text.index("--save-as")


def test_compat_steps_make_then_libresprite_then_compare():
    steps = [s["run"] for s in _load()["jobs"]["libresprite"]["steps"] if "run" in s]
    make = next(i for i, s in enumerate(steps) if "aseprite_compat.py make" in s)
    render = next(i for i, s in enumerate(steps) if " -b " in s and "--save-as" in s)
    comp = next(i for i, s in enumerate(steps) if "aseprite_compat.py compare" in s)
    assert make < render < comp


def test_compare_ignores_rgb_where_ours_is_transparent():
    ours = _img([[(0, 0, 0, 0)]])
    theirs = _img([[(255, 0, 77, 0)]])
    assert compat.compare_images(ours, theirs) == []


def test_compare_reports_opaque_pixel_difference(tmp_path, capsys):
    ours = _img([[(1, 2, 3, 255), (9, 9, 9, 255)]])
    theirs = _img([[(1, 2, 3, 255), (9, 9, 10, 255)]])
    lines = compat.compare_images(ours, theirs)
    assert lines[0].startswith("x=1 y=0 ")
    assert lines[-1] == "1 pixels differ"
    (tmp_path / "c.aseprite").write_bytes(b"")
    ours.save(tmp_path / "c.png")
    theirs.save(tmp_path / "c.libresprite.png")
    assert compat.compare(tmp_path) == 1
    assert "aseprite-compat: c: MISMATCH" in capsys.readouterr().out


def test_compare_fails_when_theirs_is_opaque_where_ours_is_transparent():
    ours = _img([[(0, 0, 0, 0)]])
    theirs = _img([[(0, 0, 0, 255)]])
    assert compat.compare_images(ours, theirs)[-1] == "1 pixels differ"


def test_compare_fails_on_missing_or_resized_render(tmp_path, capsys):
    (tmp_path / "c.aseprite").write_bytes(b"")
    _img([[(1, 2, 3, 255)]]).save(tmp_path / "c.png")
    assert compat.compare(tmp_path) == 1
    assert "missing render" in capsys.readouterr().out
    _img([[(1, 2, 3, 255), (1, 2, 3, 255)]]).save(tmp_path / "c.libresprite.png")
    assert compat.compare(tmp_path) == 1
    assert compat.compare_images(
        _img([[(1, 2, 3, 255)]]), _img([[(1, 2, 3, 255), (1, 2, 3, 255)]])
    )[0].startswith("size ")


def test_make_writes_case_set_from_the_shipped_writer(tmp_path):
    lines = compat.make(tmp_path)
    assert len(lines) == 3
    assert len(list(tmp_path.glob("*.aseprite"))) == 3
    assert len(list(tmp_path.glob("*.png"))) == 3
    for name, _, _ in compat.CASES:
        grid, palette = compat.build_case(name)
        n = palette.shape[0]
        ase = ao.parse_aseprite((tmp_path / f"{name}.aseprite").read_bytes())
        expected = np.where(grid < 0, n, grid).tolist()
        assert ao.composite_indices(ase, 0) == expected
        with Image.open(tmp_path / f"{name}.png") as png:
            assert np.array_equal(np.asarray(png.convert("RGBA")), grid_to_rgba(grid, palette))


def test_make_pico8_case_keeps_index_zero_opaque_and_transparent_at_n(tmp_path):
    compat.make(tmp_path)
    grid, palette = compat.build_case("pico8-16x16")
    ase = ao.parse_aseprite((tmp_path / "pico8-16x16.aseprite").read_bytes())
    assert ase.transparent_index == 16 == palette.shape[0]
    assert (grid == 0).any()
    with Image.open(tmp_path / "pico8-16x16.png") as png:
        pixels = np.asarray(png.convert("RGBA"))
    zero = pixels[grid == 0]
    assert len(zero) and (zero == (0, 0, 0, 255)).all()
    assert (pixels[grid == -1][:, 3] == 0).all()


def test_compat_script_imports_stay_inside_allowlist():
    allowed = {"numpy", "PIL", "pixelgen.aseprite", "pixelgen.palettes", "pixelgen.pixelate"}
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            names.append(node.module)
    assert names
    for name in names:
        assert name.split(".")[0] in sys.stdlib_module_names or name in allowed, name
