"""PXG-44 Part A: ``scripts/check_examples.py`` run-record version 2, pair rules and ``sk-or-``.

Offline. A comparison pair is written by the real CLI (``cli.app`` through ``CliRunner``) with a
``FakeLLM`` behind ``cli.make_llm`` and a ``FakeTransport`` behind ``cli.make_transport``, then a
version 2 ``run.json`` is derived from the produced sidecar. Every error case changes only the
fields it names; "recomputed" means set to the value the D3 formula gives. Key-like fixtures are
built at run time and are obviously synthetic.
"""

import ast
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from fakes import FakeLLM, FakeTransport
from openrouter_fixture import images_reply, png_b64, sprite_on_backdrop
from PIL import Image, PngImagePlugin
from typer.testing import CliRunner

import pixelgen.cli as cli

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "check_examples.py"
PAIR = "cmp-apple-16"
LLM = f"{PAIR}-llm-grid"
OR = f"{PAIR}-openrouter"
P = f"pair {LLM}/{OR}"
CAPS = {"llm-grid": (1.03104, 1.05), "openrouter": (0.92477, 0.96)}
MEMBERSHIP = f"pair {PAIR} needs exactly one llm-grid and one openrouter slug"
# Built at run time and obviously synthetic: no real key shape sits in this file.
SENTINEL = "sk" + "-or-v1-pxg44-" + uuid.uuid4().hex
HEX_TAIL = SENTINEL.rsplit("-", 1)[1]


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_examples_pxg44_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_examples_pxg44_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


checker = _load_checker()


def run_checker(root, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(root), *map(str, args)],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )


# --------------------------------------------------------------------------- builder


def _grid(size):
    return [[1 + (x + y) % 4 for x in range(size)] for y in range(size)]


def make_slug(root, slug, *, backend, prompt="a red apple", size=16):
    """Run the real CLI for one comparison member; return its parsed sidecar."""
    script = ({}, {"grid": _grid(size)}) if backend == "llm-grid" else ({},)
    reply = images_reply(png_b64(sprite_on_backdrop(7)))
    out = Path(root) / slug / f"{slug}.png"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", "dummy-anthropic")
        mp.setenv("OPENROUTER_API_KEY", "dummy-openrouter")
        mp.setattr(cli, "make_llm", lambda: FakeLLM(*script))
        mp.setattr(cli, "make_transport", lambda: FakeTransport(reply))
        args = ["generate", prompt, "--size", str(size), "--palette", "pico8"]
        args += ["--backend", backend, "-o", str(out)]
        result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == 0, (result.stdout, result.stderr)
    return json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))


def tokens_cost(line):
    prices = line["prices_usd_per_mtok"]
    raw = (
        line["input_tokens"] * prices["input_tokens"]
        + line["output_tokens"] * prices["output_tokens"]
    )
    return round(raw / 1e6, 6)


def images_cost(line):
    return round(
        round(line["key_usage_after_usd"] - line["key_usage_before_usd"], 6)
        * (1 + line["credit_fee_pct"] / 100),
        6,
    )


def recompute_line(line):
    line["cost_usd"] = tokens_cost(line) if line["unit"] == "tokens" else images_cost(line)


def recompute_step(step):
    step["cost_usd"] = round(sum(line["cost_usd"] for line in step["costs"]), 6)


def build_cost_lines(side, *, before, after, price, fee):
    lines = []
    for provider in dict.fromkeys(r["provider"] for r in side["usage"]):
        recs = [r for r in side["usage"] if r["provider"] == provider]
        if recs[0]["unit"] == "tokens":
            line = {
                "provider": provider,
                "unit": "tokens",
                "steps": [r["step"] for r in recs],
                "input_tokens": side["token_usage"]["input_tokens"],
                "output_tokens": side["token_usage"]["output_tokens"],
                "prices_usd_per_mtok": {"input_tokens": 2.0, "output_tokens": 10.0},
                "prices_confirmed_on": "2026-11-06",
                "cost_source": "sidecar-tokens-x-price",
            }
        else:
            calls = sum(r["calls"] for r in recs)
            metered = round(after - before, 6)
            expected = round(calls * price, 6)
            line = {
                "provider": provider,
                "unit": "images",
                "steps": [r["step"] for r in recs],
                "calls": calls,
                "key_usage_before_usd": before,
                "key_usage_after_usd": after,
                "metered_usd": metered,
                "price_per_image_usd": price,
                "expected_usd": expected,
                "cross_check": "match" if abs(metered - expected) <= 0.5 * price else "mismatch",
                "credit_fee_pct": fee,
                "credit_fee_read_on": "2026-11-06",
                "prices_confirmed_on": "2026-11-06",
                "cost_source": "openrouter-key-usage",
            }
        recompute_line(line)
        lines.append(line)
    return lines


def write_record(root, slug, pair, side, *, prompt, size, backend, before, after, approval):
    flag = f"--backend {backend}"
    command = (
        f'pixelgen generate "{prompt}" --size {size} --palette pico8 {flag} '
        f"-o examples/{slug}/{slug}.png"
    )
    worst, cap = CAPS[backend]
    step = {
        "kind": "generate",
        "approval": approval,
        "command": command,
        "prompt": prompt,
        "backend": side["backend"],
        "model": side["model"],
        "provider": side["provider"],
        "provider_model": side["provider_model"],
        "pixelgen_version": side["pixelgen_version"],
        "ran_at": "2026-11-06T12:00:00Z",
        "sidecar": f"{slug}.json",
        "usage_records": side["usage"],
        "costs": build_cost_lines(side, before=before, after=after, price=0.0067, fee=5.0),
        "estimate_usd": 0.0097,
        "worst_usd": worst,
        "cap_usd": cap,
        "notes": "",
    }
    recompute_step(step)
    record = {"run_record_version": 2, "slug": slug, "pair": pair, "steps": [step]}
    save_record(root, slug, record)


def load_record(root, slug):
    return json.loads((Path(root) / slug / "run.json").read_text(encoding="utf-8"))


def save_record(root, slug, record):
    (Path(root) / slug / "run.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )


def write_readme(root):
    names = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    (Path(root) / "README.md").write_text(
        "\n".join(f"- {n}" for n in names) + "\n", encoding="utf-8"
    )


def make_pair(
    root, *, pair=PAIR, edit=None, variants=None, before=1.0, after=1.0067, approval="calibration"
):
    """Two real-CLI runs (one per arm), their v2 records, a README, then ``edit``."""
    root = Path(root)
    variants = variants or {}
    for arm in ("llm-grid", "openrouter"):
        slug = f"{pair}-{arm}"
        want = {"prompt": "a red apple", "size": 16, "backend": arm, **variants.get(arm, {})}
        side = make_slug(root, slug, **want)
        write_record(root, slug, pair, side, before=before, after=after, approval=approval, **want)
    write_readme(root)
    if edit is not None:
        edit(root)
    return root


def edit_step(slug, fn):
    def _edit(root):
        record = load_record(root, slug)
        fn(record["steps"][0], record)
        save_record(root, slug, record)

    return _edit


def line_of(step, unit):
    return next(x for x in step["costs"] if x["unit"] == unit)


def anth(step):
    return line_of(step, "tokens")


def orr(step):
    return line_of(step, "images")


def set_after(step, after, *, cross):
    line = orr(step)
    line["key_usage_after_usd"] = after
    line["metered_usd"] = round(after - line["key_usage_before_usd"], 6)
    line["cross_check"] = cross
    recompute_line(line)
    recompute_step(step)


# --------------------------------------------------------------------------- assertions


def _details(proc):
    return [ln for ln in proc.stdout.splitlines() if ln.startswith("FAIL ")][:-1]


def assert_one_fail(root, relpath, reason, *extra):
    proc = run_checker(root, *extra)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    details = _details(proc)
    assert len(details) == 1, proc.stdout
    assert details[0].startswith(f"FAIL {relpath}: "), details[0]
    assert reason in details[0], details[0]
    assert proc.stdout.splitlines()[-1] == f"FAIL {root}: 1 problem(s)", proc.stdout
    found = checker.check_tree(Path(root), *(int(a) for a in extra[1:]))
    assert len(found) == 1 and reason in found[0].reason and found[0].relpath == relpath, found


def assert_clean(root, *extra):
    proc = run_checker(root, *extra)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith(f"OK {root}: "), proc.stdout
    assert checker.check_tree(Path(root)) == []


OR_RUN = f"{OR}/run.json"

# --------------------------------------------------------------------------- AC2


def test_v2_pair_from_real_cli_passes(tmp_path):
    root = make_pair(tmp_path / "examples")
    assert checker.check_tree(root) == []
    proc = run_checker(root)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == f"OK {root}: 2 runs, 2 steps, 4 images, key-material hits 0"
    assert run_checker(root, "--require-runs", 2).returncode == 0
    side = json.loads((root / OR / f"{OR}.json").read_text(encoding="utf-8"))
    assert side["usage"][1]["unit"] == "images"
    assert not [k for k in _keys(side) if re.search(r"(?i)cost|price|usd", k)]


def _keys(obj):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield key
            yield from _keys(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _keys(item)


# --------------------------------------------------------------------------- AC3


def _pr_a(step, rec):
    step["usage"] = {"input_tokens": 10, "output_tokens": 20}


def _pr_b(step, rec):
    anth(step)["steps"] = ["infer_spec", "generate"]


def _pr_c(step, rec):
    step["costs"] = [anth(step)]
    recompute_step(step)


def _pr_d(step, rec):
    anth(step)["output_tokens"] += 1
    recompute_line(anth(step))
    recompute_step(step)


def _pr_e(step, rec):
    step["usage_records"][1]["calls"] = 2


PER_RECORD = {
    "a_whole_run_usage_key": (_pr_a, "v2 step must not carry a whole-run usage figure"),
    "b_steps_set_to_both": (_pr_b, "cost line steps differ from usage records"),
    "c_openrouter_line_deleted": (_pr_c, "one cost line per provider"),
    "d_output_tokens_plus_one": (_pr_d, "tokens line differs from sidecar token counts"),
    "e_usage_records_falsified": (_pr_e, "usage_records differ from sidecar usage"),
}


@pytest.mark.parametrize("case", list(PER_RECORD))
def test_v2_per_record_pricing(tmp_path, case):
    fn, reason = PER_RECORD[case]
    root = make_pair(tmp_path / "examples", edit=edit_step(OR, fn))
    assert_one_fail(root, OR_RUN, reason)


# --------------------------------------------------------------------------- AC4


def _tl_a(step, rec):
    anth(step)["cost_usd"] = round(anth(step)["cost_usd"] + 0.000002, 6)
    recompute_step(step)


def _tl_b(step, rec):
    del anth(step)["prices_usd_per_mtok"]["output_tokens"]


def _tl_c(step, rec):
    anth(step)["cost_source"] = "openrouter-key-usage"


TOKENS_LINE = {
    "a_cost_off_formula": (_tl_a, "tokens line cost_usd off formula"),
    "b_output_price_removed": (_tl_b, "tokens line needs input_tokens and output_tokens prices"),
    "c_wrong_cost_source": (_tl_c, "tokens line cost_source must be sidecar-tokens-x-price"),
}


@pytest.mark.parametrize("case", list(TOKENS_LINE))
def test_v2_tokens_line(tmp_path, case):
    fn, reason = TOKENS_LINE[case]
    root = make_pair(tmp_path / "examples", edit=edit_step(OR, fn))
    assert_one_fail(root, OR_RUN, reason)


# --------------------------------------------------------------------------- AC5


def _ku_a(step, rec):
    orr(step)["metered_usd"] = 0.0068


def _ku_b(step, rec):
    set_after(step, 0.9993, cross="mismatch")


def _ku_c(step, rec):
    orr(step)["expected_usd"] = 0.0068


def _ku_d(step, rec):
    set_after(step, 1.2117, cross="match")


def _ku_e(step, rec):
    orr(step)["cost_usd"] = round(orr(step)["cost_usd"] + 0.000002, 6)
    recompute_step(step)


def _ku_f(root):
    make_pair(root, pair="cmp-zzz-16")

    def _price(step, rec):
        orr(step)["price_per_image_usd"] = 0.0068
        orr(step)["expected_usd"] = 0.0068

    edit_step("cmp-zzz-16-openrouter", _price)(root)


KEY_USAGE = {
    "a_metered_off_delta": (_ku_a, OR_RUN, "metered_usd differs from the key usage delta"),
    "b_after_below_before": (_ku_b, OR_RUN, "key_usage_after_usd below key_usage_before_usd"),
    "c_expected_off_product": (
        _ku_c,
        OR_RUN,
        "expected_usd differs from calls x price_per_image_usd",
    ),
    "d_cross_check_word_falsified": (_ku_d, OR_RUN, "cross_check differs from the recomputation"),
    "e_cost_off_fee_formula": (_ku_e, OR_RUN, "images line cost_usd off fee formula"),
}


@pytest.mark.parametrize("case", list(KEY_USAGE))
def test_v2_key_usage_line(tmp_path, case):
    fn, relpath, reason = KEY_USAGE[case]
    root = make_pair(tmp_path / "examples", edit=edit_step(OR, fn))
    assert_one_fail(root, relpath, reason)


def test_v2_key_usage_line_price_differs_across_tree(tmp_path):
    root = make_pair(tmp_path / "examples", edit=_ku_f)
    assert_one_fail(
        root, "cmp-zzz-16-openrouter/run.json", "price_per_image_usd differs across the tree"
    )


def test_v2_key_usage_line_honest_mismatch_is_not_a_problem(tmp_path):
    root = make_pair(
        tmp_path / "examples",
        edit=edit_step(OR, lambda s, r: set_after(s, 1.2117, cross="mismatch")),
    )
    assert orr(load_record(root, OR)["steps"][0])["cross_check"] == "mismatch"
    assert_clean(root)


# --------------------------------------------------------------------------- AC6


def _sc_a(step, rec):
    step["cost_usd"] = round(step["cost_usd"] + 0.000002, 6)


def _sc_b(step, rec):
    step["cap_usd"], step["worst_usd"] = 0.005, 0.004


def _sc_c(step, rec):
    step["worst_usd"] = 0.97


def _sc_d(step, rec):
    step["approval"] = "maybe"


STEP_CAPS = {
    "a_step_total_off": (_sc_a, "step cost_usd differs from the sum of its lines"),
    "b_cost_above_cap": (_sc_b, "cost_usd above cap_usd"),
    "c_cap_below_worst": (_sc_c, "cap_usd below worst_usd"),
    "d_approval": (_sc_d, "approval must be calibration or spend"),
}


@pytest.mark.parametrize("case", list(STEP_CAPS))
def test_v2_step_total_and_caps(tmp_path, case):
    fn, reason = STEP_CAPS[case]
    root = make_pair(tmp_path / "examples", edit=edit_step(OR, fn))
    assert_one_fail(root, OR_RUN, reason)


# --------------------------------------------------------------------------- AC7


def _command_edit(old, new):
    def _fn(step, rec):
        assert old in step["command"]
        step["command"] = step["command"].replace(old, new)

    return _fn


def _command_append(text):
    def _fn(step, rec):
        step["command"] += text

    return _fn


OUT = f"examples/{OR}/{OR}.png"
PAIR_RULES = {
    "a_size": ({"variants": {"openrouter": {"size": 32}}}, f"{P} differs in size"),
    "b_prompt": (
        {"variants": {"openrouter": {"prompt": "a green apple"}}},
        f"{P} differs in prompt",
    ),
    "c_prompt_and_size": (
        {"variants": {"openrouter": {"size": 32, "prompt": "a green apple"}}},
        f"{P} differs in prompt,size",
    ),
    "d_llm_grid_removed": ({"edit": lambda root: shutil.rmtree(root / LLM)}, MEMBERSHIP),
    "e_two_llm_grid": ({"variants": {"openrouter": {"backend": "llm-grid"}}}, MEMBERSHIP),
    "f_output_path": (
        {"edit": edit_step(OR, _command_edit(OUT, f"/tmp/x/{OR}.png"))},
        f"{OR} -o is not examples/{OR}/{OR}.png",
    ),
    "g_backend_flag": (
        {"edit": edit_step(OR, _command_edit("--backend openrouter", "--backend llm-grid"))},
        f"{OR} --backend differs from sidecar backend",
    ),
    "h_extra_flag": (
        {"edit": edit_step(OR, _command_append(" --outline"))},
        f"{P} differs in flags",
    ),
}


@pytest.mark.parametrize("case", list(PAIR_RULES))
def test_v2_pair_rules(tmp_path, case):
    kwargs, reason = PAIR_RULES[case]
    root = make_pair(tmp_path / "examples", **kwargs)
    assert_one_fail(root, OR_RUN, reason)


# --------------------------------------------------------------------------- AC8


def _to_v1_sidecar(root):
    path = root / OR / f"{OR}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = 1
    for key in ("provider", "provider_model", "usage", "dropped_fields"):
        del data[key]
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def test_v2_versions_v2_record_over_v1_sidecar(tmp_path):
    root = make_pair(tmp_path / "examples", edit=_to_v1_sidecar)
    assert_one_fail(root, OR_RUN, "run record v2 needs a v2 sidecar")


def test_v2_versions_unknown_version(tmp_path):
    def _v3(step, rec):
        rec["run_record_version"] = 3

    root = make_pair(tmp_path / "examples", edit=edit_step(OR, _v3))
    assert_one_fail(root, OR_RUN, "run_record_version must be 1 or 2")


def test_v2_versions_v1_record_over_v2_sidecar_passes(tmp_path):
    root = tmp_path / "examples"
    side = make_slug(root, "red-apple", backend="llm-grid")
    tokens = side["token_usage"]
    step = {
        "kind": "generate",
        "command": (
            'pixelgen generate "a red apple" --size 16 --palette pico8 '
            "-o examples/red-apple/red-apple.png"
        ),
        "prompt": "a red apple",
        "model": side["model"],
        "pixelgen_version": side["pixelgen_version"],
        "ran_at": "2026-11-06T12:00:00Z",
        "sidecar": "red-apple.json",
        "usage": {"input_tokens": tokens["input_tokens"], "output_tokens": tokens["output_tokens"]},
        "prices_usd_per_mtok": {"input_tokens": 2.0, "output_tokens": 10.0},
        "prices_confirmed_on": "2026-11-06",
        "estimate_usd": 0.0005,
        "cost_usd": round((20 * 2.0 + 40 * 10.0) / 1e6, 6),
        "cap_usd": 1.0,
        "notes": "",
    }
    record = {"run_record_version": 1, "slug": "red-apple", "steps": [step]}
    save_record(root, "red-apple", record)
    assert json.loads((root / "red-apple" / "red-apple.json").read_text())["schema_version"] == 2
    assert_clean(root)


# --------------------------------------------------------------------------- AC9


def _notes(step, rec):
    step["notes"] = SENTINEL


def _escaped_notes(root):
    record = load_record(root, OR)
    record["steps"][0]["notes"] = SENTINEL
    text = json.dumps(record, indent=2).replace("sk-or-", "sk\\u002dor-")
    assert "sk-or-" not in text
    (root / OR / "run.json").write_text(text + "\n", encoding="utf-8")


def _png_text(zip_):
    def _edit(root):
        path = root / OR / f"{OR}.png"
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", SENTINEL, zip=zip_)
        with Image.open(path) as image:
            image.load()
            pixels = image.copy()
        pixels.save(path, format="PNG", pnginfo=info)

    return _edit


def _dir_name(root):
    (root / OR).rename(root / SENTINEL)


def _command_variable(root):
    for slug in (LLM, OR):
        record = load_record(root, slug)
        del record["pair"]
        if slug == OR:
            record["steps"][0]["command"] += " --note OPENROUTER_API_KEY=dummy"
        save_record(root, slug, record)


OR_KEY_PLACES = {
    "notes": edit_step(OR, _notes),
    "png_text_chunk": _png_text(False),
    "png_ztxt_chunk": _png_text(True),
    "json_escaped_note": _escaped_notes,
    "slug_dir_name": _dir_name,
    "command_variable": _command_variable,
}


@pytest.mark.parametrize("where", list(OR_KEY_PLACES))
def test_or_key_material(tmp_path, where):
    root = make_pair(tmp_path / "examples", edit=OR_KEY_PLACES[where])
    proc = run_checker(root)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    out = proc.stdout + proc.stderr
    assert "sk-or-" not in out and HEX_TAIL not in out
    details = _details(proc)
    if where == "slug_dir_name":
        assert len(details) >= 1
        assert any("key-like material" in ln for ln in details), proc.stdout
    elif where == "command_variable":
        assert len(details) == 1 and OR_RUN in details[0], proc.stdout
        assert "command contains key material or an API key variable" in details[0]
    else:
        assert len(details) == 1, proc.stdout
        assert "key-like material" in details[0], details[0]


# --------------------------------------------------------------------------- AC10


def test_v2_billed_runs(tmp_path):
    good = make_pair(tmp_path / "good")
    assert run_checker(good, "--require-runs", 2).returncode == 0

    zero = make_pair(tmp_path / "zero", after=1.0)
    step = load_record(zero, OR)["steps"][0]
    assert orr(step)["metered_usd"] == 0 and orr(step)["cross_check"] == "mismatch"
    assert checker.check_tree(zero) == []
    assert_clean(zero)
    assert_one_fail(zero, ".", "1 run(s) with a billed first generate step", "--require-runs", 2)


# --------------------------------------------------------------------------- AC11


def _imports(source):
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * node.level) + (node.module or "")
            names.append(base)
            names.extend(f"{base}.{alias.name}" for alias in node.names)
    return names


OFFLINE_BANNED = ("urllib.request", "http", "socket", "pixelgen.providers", "pixelgen.backends")


def test_checker_v2_imports_offline():
    names = _imports(SCRIPT.read_text(encoding="utf-8"))
    assert "pixelgen.export" in names and "shlex" in names
    hits = [n for n in names for b in OFFLINE_BANNED if n == b or n.startswith(b + ".")]
    assert hits == [], hits
