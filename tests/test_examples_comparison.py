"""PXG-44 Part C: the committed live llm-grid vs openrouter comparison under ``examples/``.

Offline and read-only. The twelve ``examples/cmp-*`` slugs were produced by the operator's live run
(Part B); these tests pin what the committed tree must keep saying: the checker accepts it, the
pairs are complete, no sidecar carries a money field, the README records agree with every
``run.json``, and the PXG-14 artefacts are still byte-identical.
"""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXAMPLES = REPO / "examples"
SCRIPT = REPO / "scripts" / "check_examples.py"
OK_LINE = "OK examples: 15 runs, 16 steps, 32 images, key-material hits 0"
SECTION = "## Provider comparison (PXG-44)"
MONEY_KEY = re.compile(r"(?i)cost|price|usd")
RESULTS = {"CONFIRMED", "CORRECTED", "NOT OBSERVED"}
# GR-006: the 15 PXG-14 artefact files, sha256 at the PXG-14 base (frozen by PXG-44 AC20).
PXG14_MANIFEST = """\
db464c3520299860c370f3bff9d243b83adcf13babdb4c570f6985edd1938a00  knight/knight-v2.json
bc6aaffbdfd9910b808df15928621e601490acf5422a7a313fe6e854b4363ef7  knight/knight-v2.png
c9df0e64245d571f81cff2fd15c6ff3f3b2cc5627de6870285a70164f05d88a9  knight/knight-v2@8x.png
f6bea8c175cf0e0864074c796861f939cb3ef804198c3d1f656d44462e180741  knight/knight.json
8467c4dfca294902366423a6dd08e0e6809a6e9957f2d04eb8888f64cc9413c7  knight/knight.png
ae6d2efeddd44e6f50f309b0448a36b3363e1f37c2efa367141831e7f8df501d  knight/knight@8x.png
a8f4bccab18b9eaa9e12b9d3c93febef283c6cac192b07d0fbe063ad56bbafa3  knight/run.json
970bb21ff844ef90dae3dc6b009c398636c28acb3d65d6f4601c6652f9be0527  red-apple/red-apple.json
58e84f57eb5d93a550319d2725dd6223d1360f7c504dd87559e5c25108537a3d  red-apple/red-apple.png
bc8fd1f2aa5986577590c5b020ebecf84ba60bbb64506afdb6bbe0bf326736f6  red-apple/red-apple@8x.png
4054eb8cbb00b0d166c3d553332e75843bdb93c81361f8ae8d78205d1a541a9d  red-apple/run.json
75f49387762959fb9f99d2efcfd5924c5d1c0218d378585ec9528eb50ca2ac77  slime/run.json
61ec7d33277f6f5c655f10e30c74e6bf02fe35b29e4f18f2a033a5b1b89249c6  slime/slime.json
fa8b2a1845196f22ee5eb25441bf0d56f3f82ffb2747943bc354c5fd261feb36  slime/slime.png
34e3a02e9d91f0d0bc212c264c79a57d64bfa3ae2ada8f9b22b3a83a951494ca  slime/slime@8x.png
"""
PXG14_FROZEN = {
    line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in PXG14_MANIFEST.splitlines()
}


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_examples_pxg44_c", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["check_examples_pxg44_c"] = mod
    spec.loader.exec_module(mod)
    return mod


checker = _load_checker()


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _slugs():
    return sorted(p.name for p in EXAMPLES.glob("cmp-*") if p.is_dir())


def _step(slug):
    return _json(EXAMPLES / slug / "run.json")["steps"][0]


def _money_keys(obj, path=""):
    hits = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if MONEY_KEY.search(key):
                hits.append(f"{path}/{key}")
            hits += _money_keys(value, f"{path}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            hits += _money_keys(value, f"{path}[{i}]")
    return hits


def _readme_section():
    text = (EXAMPLES / "README.md").read_text(encoding="utf-8")
    assert text.count(SECTION) == 1, "README needs exactly one comparison section"
    return text[text.index(SECTION) :]


def _subsection(section, heading):
    start = section.index(f"### {heading}\n")
    body = section[start:].split("\n", 1)[1]
    stop = re.search(r"^#{2,3} ", body, re.M)
    return body[: stop.start()] if stop else body


def _rows(block):
    return [
        [c.strip() for c in line.strip().strip("|").split("|")]
        for line in block.splitlines()
        if line.startswith("| ") and not line.startswith("| ---") and not line.startswith("|--")
    ]


def test_committed_comparison_tree():
    slugs = _slugs()
    assert len(slugs) == 12
    pairs = {}
    for slug in slugs:
        record = _json(EXAMPLES / slug / "run.json")
        assert record["run_record_version"] == 2, slug
        sidecar = _json(EXAMPLES / slug / f"{slug}.json")
        assert sidecar["schema_version"] == 2, slug
        assert _money_keys(sidecar) == [], slug
        pairs.setdefault(record["pair"], set()).add(record["steps"][0]["backend"])
    assert len(pairs) == 6
    assert all(backends == {"llm-grid", "openrouter"} for backends in pairs.values()), pairs
    assert checker.check_tree(EXAMPLES, require_runs=15) == []


def test_committed_comparison_tree_checker_prints_the_ok_line():
    proc = subprocess.run(
        [sys.executable, "scripts/check_examples.py", "examples", "--require-runs", "15"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == OK_LINE


def test_pxg14_artefacts_are_frozen():
    for rel, digest in PXG14_FROZEN.items():
        assert hashlib.sha256((EXAMPLES / rel).read_bytes()).hexdigest() == digest, rel


def test_comparison_readme_records():
    section = _readme_section()
    approvals = {
        prefix: [line for line in section.splitlines() if line.startswith(prefix)]
        for prefix in ("PXG-44 CALIBRATION APPROVED ", "PXG-44 SPEND APPROVED ")
    }
    for prefix, found in approvals.items():
        assert len(found) == 1, prefix
    per_image = {
        float(re.search(r"price per image \$([0-9.]+)", line).group(1))
        for found in approvals.values()
        for line in found
    }
    assert len(per_image) == 1, "the two approval lines differ in price per image"
    (price,) = per_image

    slugs = _slugs()
    rows = {r[0].strip("`"): r for r in _rows(_subsection(section, "Runs")) if r[0].startswith("`")}
    assert sorted(rows) == slugs
    for slug in slugs:
        step = _step(slug)
        assert float(rows[slug][6]) == step["cost_usd"], f"{slug}: cost cell differs from run.json"
        for line in step["costs"]:
            if line["unit"] == "images":
                assert line["price_per_image_usd"] == price, f"{slug}: price per image"

    failed = _subsection(section, "Failed runs")
    failed_total = 0.0
    if "No run failed." not in failed:
        table = _rows(failed)
        column = table[0].index("Cost (USD)")
        failed_total = sum(float(r[column]) for r in table[1:])
    total = re.findall(r"^Total spent: ([0-9.]+) USD$", section, re.M)
    assert len(total) == 1
    expected = round(sum(_step(s)["cost_usd"] for s in slugs) + failed_total, 6)
    assert float(total[0]) == expected, f"Total spent {total[0]} != {expected}"

    items = [r for r in _rows(_subsection(section, "PXG-42 open items")) if r[0].isdigit()]
    assert [r[0] for r in items] == ["1", "2", "3", "4", "5", "6"]
    assert all(r[2] in RESULTS for r in items), [r[2] for r in items]
