"""Supply-chain structure of .github/ (PXG-46): lock-driven installs, pins, credentials.

Reads workflow YAML, ``uv.lock`` and ``dependabot.yml`` only (stdlib ``tomllib`` + PyYAML). Nothing
here starts a child process, touches the network, or looks at ``sys.version_info``.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
CI = WORKFLOWS / "ci.yml"
COMPAT = WORKFLOWS / "aseprite-compat.yml"
DEPENDABOT = ROOT / ".github" / "dependabot.yml"
LOCK = ROOT / "uv.lock"

SETUP_PY_EXPR = "${{ steps.setup-python.outputs.python-path }}"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _locked_jobs() -> dict[str, dict]:
    return {
        "ci.test": _yaml(CI)["jobs"]["test"],
        "ci.lint": _yaml(CI)["jobs"]["lint"],
        "compat.libresprite": _yaml(COMPAT)["jobs"]["libresprite"],
    }


def _all_jobs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in (CI, COMPAT):
        tag = "ci" if path == CI else "compat"
        for name, job in _yaml(path)["jobs"].items():
            out[f"{tag}.{name}"] = job
    return out


def _run_text(job: dict) -> str:
    return "\n".join(s["run"] for s in job.get("steps", []) if "run" in s)


def _install_steps(job: dict) -> list[dict]:
    return [s for s in job["steps"] if str(s.get("name", "")).startswith("Install")]


def test_locked_jobs_sync_from_the_lock_only():
    for jid, job in _locked_jobs().items():
        installs = _install_steps(job)
        assert len(installs) == 1, f"{jid}: expected one Install* step, got {len(installs)}"
        run = installs[0]["run"].strip()
        assert run.startswith("uv sync --locked"), f"{jid}: install step is {run!r}"
        text = _run_text(job)
        for bad in ("pip install", "uv pip", "--frozen", "uv lock"):
            assert bad not in text, f"{jid}: forbidden form {bad!r} in run text"
        for m in re.finditer(r"uv sync\b[^\n]*", text):
            assert "--locked" in m.group(0), f"{jid}: uv sync without --locked: {m.group(0)!r}"


def test_uv_run_never_resyncs():
    total = 0
    for jid, job in _all_jobs().items():
        for m in re.finditer(r"\buv run\b[^\n]*", _run_text(job)):
            total += 1
            assert m.group(0).startswith("uv run --no-sync "), f"{jid}: bare uv run: {m.group(0)!r}"
    assert total >= 1, "no `uv run` found in any workflow run text"


def test_uv_sync_targets_setup_python_interpreter():
    for jid, job in _locked_jobs().items():
        steps = [s for s in job["steps"] if s.get("id") == "setup-python"]
        assert len(steps) == 1, f"{jid}: needs one step with id: setup-python"
        if jid == "ci.test":
            got = (steps[0].get("with") or {}).get("python-version")
            assert got == "${{ matrix.python-version }}", f"{jid}: setup-python version {got!r}"
        for step in job["steps"]:
            if "uv sync" in step.get("run", ""):
                env = step.get("env") or {}
                assert env.get("UV_PYTHON") == SETUP_PY_EXPR, (
                    f"{jid}: step {step.get('name')!r} UV_PYTHON is {env.get('UV_PYTHON')!r}"
                )


def test_setup_uv_pins_one_exact_uv_with_checksum():
    found: dict[str, dict] = {}
    for jid, job in _all_jobs().items():
        for step in job.get("steps", []):
            uses = str(step.get("uses", ""))
            if uses.startswith("astral-sh/setup-uv"):
                assert jid not in found, f"{jid}: more than one setup-uv step"
                assert re.fullmatch(r"astral-sh/setup-uv@[0-9a-f]{40}", uses), (
                    f"{jid}: setup-uv not SHA-pinned: {uses!r}"
                )
                with_ = step.get("with") or {}
                version, checksum = with_.get("version"), with_.get("checksum")
                assert re.fullmatch(r"\d+\.\d+\.\d+", str(version)), (
                    f"{jid}: bad version {version!r}"
                )
                assert re.fullmatch(r"[0-9a-f]{64}", str(checksum)), (
                    f"{jid}: bad or missing checksum {checksum!r}"
                )
                found[jid] = {"uses": uses, "version": version, "checksum": checksum}
    assert set(found) == {"ci.test", "ci.lint", "compat.libresprite"}, (
        f"setup-uv in {sorted(found)}"
    )
    triples = {tuple(v.values()) for v in found.values()}
    assert len(triples) == 1, f"setup-uv pins differ across jobs: {found}"
    for jid, job in _all_jobs().items():
        text = _run_text(job)
        for bad in ("pip install uv", "astral.sh"):
            assert bad not in text, f"{jid}: bootstraps uv another way ({bad!r})"


def test_every_checkout_drops_persisted_credentials():
    checkouts = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for jid, job in _yaml(path)["jobs"].items():
            for step in job.get("steps", []):
                if str(step.get("uses", "")).startswith("actions/checkout@"):
                    ok = (step.get("with") or {}).get("persist-credentials") is False
                    checkouts.append((path.name, jid, ok))
    assert len(checkouts) >= 4, f"only {len(checkouts)} checkouts found"
    bad = [f"{f}:{j}" for f, j, ok in checkouts if not ok]
    good = len(checkouts) - len(bad)
    assert not bad, f"{good} of {len(checkouts)} checkouts drop credentials; missing in {bad}"


def test_install_smoke_stays_a_plain_pip_user_install():
    job = _yaml(CI)["jobs"]["install-smoke"]
    assert not [s for s in job["steps"] if "setup-uv" in str(s.get("uses", ""))]
    text = _run_text(job)
    assert not re.search(r"\buv\b", text), "install-smoke run text mentions uv"
    assert "python -m venv" in text
    assert "-m pip install ." in text


def test_no_setup_python_pip_cache_and_uv_cache_off():
    for path in (CI, COMPAT):
        for jid, job in _yaml(path)["jobs"].items():
            for step in job.get("steps", []):
                uses = str(step.get("uses", ""))
                with_ = step.get("with") or {}
                if uses.startswith("actions/setup-python@"):
                    for key in ("cache", "cache-dependency-path"):
                        assert key not in with_, f"{path.name}:{jid}: setup-python has {key!r}"
                if uses.startswith("astral-sh/setup-uv@"):
                    assert with_.get("enable-cache") is False, (
                        f"{path.name}:{jid}: setup-uv enable-cache is {with_.get('enable-cache')!r}"
                    )


def test_dependabot_tracks_actions_and_uv_lock():
    assert DEPENDABOT.is_file(), ".github/dependabot.yml is missing"
    text = DEPENDABOT.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    assert cfg["version"] == 2
    updates = cfg["updates"]
    ecosystems = [u.get("package-ecosystem") for u in updates]
    assert sorted(ecosystems) == ["github-actions", "uv"], f"ecosystems: {ecosystems}"
    for u in updates:
        eco = u["package-ecosystem"]
        assert u.get("directory") == "/", f"{eco}: directory {u.get('directory')!r}"
        assert (u.get("schedule") or {}).get("interval") == "weekly", f"{eco}: schedule"
        days = (u.get("cooldown") or {}).get("default-days")
        assert isinstance(days, int) and days >= 7, f"{eco}: cooldown.default-days is {days!r}"
        assert len(u.get("groups") or {}) == 1, f"{eco}: expected one group"
        assert "registries" not in u, f"{eco}: registries key present"
        if eco == "uv":
            assert u.get("versioning-strategy") == "lockfile-only", f"{eco}: versioning-strategy"
    assert "registries" not in cfg
    assert "secrets." not in text


_MARKER = re.compile(r"python_full_version (<|>=) '(\d+)\.(\d+)'")


def _admits(markers: list[str], leg: tuple[int, int], pkg: str) -> bool:
    if not markers:
        return True
    for marker in markers:
        m = _MARKER.fullmatch(marker)
        assert m, f"{pkg}: unsupported resolution marker {marker!r}"
        bound = (int(m.group(2)), int(m.group(3)))
        if (leg < bound) if m.group(1) == "<" else (leg >= bound):
            return True
    return False


def _wheel_admits(filename: str, leg: tuple[int, int]) -> bool:
    py_set, abi_set, plat_set = filename.removesuffix(".whl").split("-")[-3:]
    major, minor = leg
    cp = f"cp{major}{minor}"
    linux = re.compile(r"manylinux\w*_x86_64")
    plats = plat_set.split(".")
    for py in py_set.split("."):
        for abi in abi_set.split("."):
            if py in ("py3", "py2") and abi == "none":
                if "any" in plats or any(linux.fullmatch(p) for p in plats):
                    return True
            if py == cp and abi == cp and any(linux.fullmatch(p) for p in plats):
                return True
            m = re.fullmatch(rf"cp{major}(\d+)", py)
            if m and abi == "abi3" and int(m.group(1)) <= minor:
                if any(linux.fullmatch(p) for p in plats):
                    return True
    return False


def test_lock_has_an_installable_wheel_for_every_leg():
    legs = [
        tuple(int(x) for x in v.split("."))
        for v in _yaml(CI)["jobs"]["test"]["strategy"]["matrix"]["python-version"]
    ]
    assert legs
    packages = tomllib.loads(LOCK.read_text(encoding="utf-8"))["package"]
    for leg in legs:
        checked = 0
        for pkg in packages:
            if "editable" in pkg["source"]:
                continue
            label = f"{pkg['name']} {pkg['version']}"
            if not _admits(pkg.get("resolution-markers", []), leg, label):
                continue
            checked += 1
            names = [w["url"].rsplit("/", 1)[-1] for w in pkg.get("wheels", [])]
            assert any(_wheel_admits(n, leg) for n in names), (
                f"{label} has no wheel for {leg[0]}.{leg[1]}"
            )
        assert checked, f"no package admitted for {leg}"
