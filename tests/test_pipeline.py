"""Unit tests for the UI-free generate pipeline (PXG-11)."""

import ast
import re
from pathlib import Path

import numpy as np
import pytest
from fakes import FakeLLM

import pixelgen.pipeline as pipeline
from pixelgen.backends.base import GenerationUsage, create_generator
from pixelgen.errors import BackendUnavailableError, OutputExistsError, PaletteError, PixelgenError
from pixelgen.export import TokenUsage, load_sidecar
from pixelgen.llm import Usage
from pixelgen.palettes import list_palettes, load_palette
from pixelgen.pixelate import grid_to_rgba
from pixelgen.spec import ArtSpec


def _grid(seed: int = 0, side: int = 16) -> list[list[int]]:
    return [
        [
            -1 if r in (0, side - 1) or c in (0, side - 1) else (r * 3 + c * 5 + seed) % 16
            for c in range(side)
        ]
        for r in range(side)
    ]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("a red apple", "a-red-apple"),
        ("  Hello, World!! ", "hello-world"),
        ("Café crème", "cafe-creme"),
        ("../../etc/passwd", "etc-passwd"),
        ("sprite v2", "sprite-ver2"),
        ("🍎🍎", "pixelgen"),
        ("", "pixelgen"),
        ("a" * 200, "a" * 48),
        ("a" * 47 + "!b", "a" * 47),
    ],
)
def test_slug_table(prompt, expected):
    slug = pipeline.slugify_prompt(prompt)
    assert slug == expected
    assert re.fullmatch(r"[a-z0-9-]{1,50}", slug)


def test_next_free_output_returns_the_plain_slug_when_nothing_exists(tmp_path):
    assert pipeline.next_free_output(tmp_path, "x", 8) == tmp_path / "x"


def test_next_free_output_skips_any_existing_target(tmp_path):
    (tmp_path / "x@8x.png").write_bytes(b"orphan")
    assert pipeline.next_free_output(tmp_path, "x", 8) == tmp_path / "x_2"
    (tmp_path / "x_2.json").write_text("{}")
    assert pipeline.next_free_output(tmp_path, "x", 8) == tmp_path / "x_3"


def test_palette_name_unknown_lists_available():
    with pytest.raises(PaletteError) as err:
        pipeline.ensure_palette_known("nope")
    for name in list_palettes():
        assert name in str(err.value)
    assert len(list_palettes()) == 5


def test_palette_name_known_passes():
    pipeline.ensure_palette_known("pico8")


def test_to_token_usage_sums_by_attribute_not_type():
    total = pipeline.to_token_usage(Usage(5, 7), GenerationUsage(3, 100, 200))
    assert total == TokenUsage(input_tokens=105, output_tokens=207, calls=4)


def test_build_spec_layers_flag_over_inferred_over_preset():
    llm = FakeLLM({"dither": "floyd-steinberg", "palette": "nes"})
    spec, inferred = pipeline.build_spec("p", "gameboy", {"palette": "pico8"}, llm)
    assert (spec.palette, spec.dither, spec.outline) == ("pico8", "floyd-steinberg", False)
    assert inferred is not None and llm.call_count == 1


def test_build_spec_without_llm_makes_no_call_and_rejects_unknown_palette():
    spec, inferred = pipeline.build_spec("p", None, {"palette": None})
    assert inferred is None and spec.subject == "p"
    with pytest.raises(PaletteError):
        pipeline.build_spec("p", None, {"palette": "nope"})


def _run(tmp_path, grid, **kwargs):
    llm = FakeLLM({}, {"grid": grid})
    params = dict(
        llm=llm,
        generator=create_generator("llm-grid", llm=llm),
        backend_name="llm-grid",
        preset_name=None,
        cli_overrides={"width": 16, "height": 16, "palette": "pico8"},
        output=None,
        overwrite=False,
        scale=4,
        out_dir=tmp_path / "out",
    )
    params.update(kwargs)
    return pipeline.run_generate("a red apple", **params), llm


def test_run_generate_passes_backend_grid_through_pixelate(tmp_path, monkeypatch):
    calls = []
    real = pipeline.pixelate
    fake_image = real(
        grid_to_rgba(np.array(_grid()), load_palette("pico8")),
        ArtSpec(subject="x", width=16, height=16),
        load_palette("pico8"),
    )

    def spy(img, spec, palette, **kw):
        calls.append(img.copy())
        return fake_image

    exported = []
    real_export = pipeline.export

    def export_spy(output, rgba, **kw):
        exported.append(rgba)
        return real_export(output, rgba, **kw)

    monkeypatch.setattr(pipeline, "pixelate", spy)
    monkeypatch.setattr(pipeline, "export", export_spy)
    grid = _grid()
    result, llm = _run(tmp_path, grid)
    assert len(calls) == 1
    assert np.array_equal(calls[0], grid_to_rgba(np.array(grid), load_palette("pico8")))
    assert exported[0] is fake_image
    assert llm.call_count == 2
    sidecar = load_sidecar(result.export.sidecar_path)
    assert sidecar.token_usage == TokenUsage(input_tokens=20, output_tokens=40, calls=2)
    assert sidecar.backend == "llm-grid" and sidecar.model == "fake-model"


def test_render_grid_writes_outputs_from_a_grid(tmp_path):
    palette = load_palette("pico8")
    spec = ArtSpec(subject="x", width=16, height=16)
    result = pipeline.render_grid(
        spec,
        _grid(),
        palette,
        output=tmp_path / "solo",
        scale=2,
        overwrite=False,
        backend="llm-grid",
        model=None,
        token_usage=TokenUsage(),
    )
    for path in (result.export.png_path, result.export.preview_path, result.export.sidecar_path):
        assert path.is_file()
    assert result.image.shape == (16, 16, 4)


def test_render_grid_passes_overwrite_through(tmp_path):
    palette = load_palette("pico8")
    spec = ArtSpec(subject="x", width=16, height=16)
    kwargs = dict(
        output=tmp_path / "solo",
        scale=2,
        backend="llm-grid",
        model=None,
        token_usage=TokenUsage(),
    )
    pipeline.render_grid(spec, _grid(), palette, overwrite=False, **kwargs)
    with pytest.raises(OutputExistsError):
        pipeline.render_grid(spec, _grid(1), palette, overwrite=False, **kwargs)
    pipeline.render_grid(spec, _grid(1), palette, overwrite=True, **kwargs)


def test_run_generate_default_output_never_overwrites(tmp_path):
    first, _ = _run(tmp_path, _grid(0), overwrite=True)
    second, _ = _run(tmp_path, _grid(1), overwrite=True)
    assert first.export.png_path.name == "a-red-apple.png"
    assert second.export.png_path.name == "a-red-apple_2.png"


def test_on_spec_runs_before_generation(tmp_path):
    seen = []
    llm = FakeLLM({}, lambda call: seen.append(llm.call_count) or {"grid": _grid()})
    pipeline.run_generate(
        "a red apple",
        llm=llm,
        generator=create_generator("llm-grid", llm=llm),
        backend_name="llm-grid",
        preset_name=None,
        cli_overrides={"width": 16, "height": 16},
        output=tmp_path / "o",
        overwrite=True,
        scale=2,
        on_spec=lambda spec: seen.append(("spec", spec.width)),
    )
    assert seen == [("spec", 16), 2]


def test_pipeline_is_ui_free():
    tree = ast.parse(Path(pipeline.__file__).read_text())
    banned = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        banned += [n for n in names if n.split(".")[0] in ("typer", "rich") or n == "pixelgen.cli"]
    assert banned == []


# --- X-PXG-38 fix-forward 1: the UI-free preflight ----------------------------------------


def test_preflight_generate_resolves_the_default_name_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "home"))
    out = tmp_path / "out"
    out.mkdir()
    (out / "a-red-apple.png").write_bytes(b"x")
    before = sorted(tmp_path.rglob("*"))
    got = pipeline.preflight_generate("a red apple", None, 4, out_dir=out)
    assert got == out / "a-red-apple_2"
    assert sorted(tmp_path.rglob("*")) == before


def test_preflight_generate_missing_directories_are_fine_when_an_ancestor_is_writable(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "deep" / "home"))
    target = tmp_path / "new" / "dir" / "x.png"
    assert pipeline.preflight_generate("p", target, 4) == target
    assert not (tmp_path / "new").exists()


def test_preflight_generate_refuses_a_file_ancestor(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "home"))
    (tmp_path / "afile").write_text("x")
    with pytest.raises(PixelgenError, match=r"afile.*not a directory"):
        pipeline.preflight_generate("p", tmp_path / "afile" / "sub" / "x.png", 4)


def test_preflight_generate_refuses_a_home_that_is_a_file(tmp_path, monkeypatch):
    home = tmp_path / "homefile"
    home.write_text("x")
    monkeypatch.setenv("PIXELGEN_HOME", str(home))
    with pytest.raises(PixelgenError, match=r"homefile.*not a directory"):
        pipeline.preflight_generate("p", tmp_path / "x.png", 4)


def test_preflight_generate_collision_exhaustion_raises_output_exists(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXELGEN_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(pipeline, "MAX_COLLISION_INDEX", 2)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("p", "p_2"):
        (out / f"{name}.png").write_bytes(b"x")
    with pytest.raises(OutputExistsError, match="no free output name"):
        pipeline.preflight_generate("p", None, 4, out_dir=out)


def test_ensure_backend_available_accepts_llm_grid_and_refuses_diffusion():
    pipeline.ensure_backend_available("llm-grid")
    with pytest.raises(BackendUnavailableError):
        pipeline.ensure_backend_available("diffusion")
