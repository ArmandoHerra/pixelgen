"""Structural assertions over .github/ (stdlib + PyYAML only)."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
GITHUB = ROOT / ".github"
CI = GITHUB / "workflows" / "ci.yml"


def _load() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


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


def test_ci_triggers_on_push_to_main_and_pull_request():
    trig = _triggers(_load())
    assert "main" in trig["push"]["branches"]
    assert "pull_request" in trig
    assert "workflow_dispatch" in trig


def test_ci_test_matrix_is_exactly_311_to_314():
    strategy = _load()["jobs"]["test"]["strategy"]
    assert strategy["matrix"]["python-version"] == ["3.11", "3.12", "3.13", "3.14"]
    assert strategy["fail-fast"] is False


def test_ci_has_no_allowed_failures():
    offenders = [n for n in _walk(_load()) if isinstance(n, dict) and "continue-on-error" in n]
    assert not offenders, "continue-on-error present"


def test_ci_runs_on_ubuntu_latest_only():
    for name, job in _load()["jobs"].items():
        assert job["runs-on"] == "ubuntu-latest", name


def test_ci_test_job_installs_dev_extra_and_runs_pytest():
    text = _run_text(_load()["jobs"]["test"])
    assert 'pip install -e ".[dev]"' in text
    assert "python -m pytest" in text


def test_install_smoke_is_non_editable_and_runs_cli():
    jobs = _load()["jobs"]
    assert "install-smoke" in jobs
    text = _run_text(jobs["install-smoke"])
    for needle in (
        "python -m venv",
        "pip install .",
        "import pixelgen",
        "check_installed_package.py",
    ):
        assert needle in text, needle
    assert re.search(r'pixelgen"? --help', text)
    assert not re.search(r"pip install\s+(-e|--editable)", text)


def test_no_anthropic_reference_under_github_dir():
    files = [p for p in GITHUB.rglob("*") if p.is_file()]
    workflows = [p for p in (GITHUB / "workflows").glob("*") if p.suffix in (".yml", ".yaml")]
    assert workflows, "no workflow file under .github/workflows/"
    offenders = [
        str(p.relative_to(ROOT))
        for p in files
        if "anthropic" in p.read_text(encoding="utf-8", errors="replace").lower()
    ]
    assert not offenders, f"anthropic referenced in: {offenders}"


def test_ci_sets_no_anthropic_env_and_uses_no_secrets():
    wf = _load()
    bad = [
        k
        for n in _walk(wf)
        if isinstance(n, dict)
        for key, env in n.items()
        if key == "env" and isinstance(env, dict)
        for k in env
        if re.search("anthropic", str(k), re.IGNORECASE)
    ]
    assert not bad, f"ANTHROPIC env keys: {bad}"
    assert "secrets." not in CI.read_text(encoding="utf-8")


def test_ci_actions_pinned_by_sha_with_version_comment():
    pattern = re.compile(r"uses:\s*[\w.-]+/[\w.-]+@[0-9a-f]{40}\s+#\s*v\d")
    lines = [ln for ln in CI.read_text(encoding="utf-8").splitlines() if "uses:" in ln]
    assert lines
    bad = [ln for ln in lines if not pattern.search(ln)]
    assert not bad, f"unpinned: {bad}"


def test_ci_permissions_read_only():
    wf = _load()
    assert wf["permissions"] == {"contents": "read"}
    with_perms = [n for n, j in wf["jobs"].items() if "permissions" in j]
    assert not with_perms, f"jobs define permissions: {with_perms}"


def test_ci_job_set_is_exactly_test_install_smoke_lint():
    assert set(_load()["jobs"]) == {"test", "install-smoke", "lint"}


def test_ci_lint_job_runs_ruff_from_pip_dev_install():
    text = _run_text(_load()["jobs"]["lint"])
    needles = ('pip install -e ".[dev]"', "ruff check .", "ruff format --check .")
    positions = []
    for needle in needles:
        assert needle in text, f"{needle!r} not in lint run text"
        positions.append(text.index(needle))
    assert positions == sorted(positions), "lint commands out of order"


def test_ci_installs_no_uv():
    wf = _load()
    setup_uv = [
        s["uses"]
        for job in wf["jobs"].values()
        for s in job.get("steps", [])
        if "setup-uv" in str(s.get("uses", ""))
    ]
    assert not setup_uv, f"setup-uv used: {setup_uv}"
    offenders = [name for name, job in wf["jobs"].items() if re.search(r"\buv\b", _run_text(job))]
    assert not offenders, f"uv in run text of jobs: {offenders}: " + "; ".join(
        _run_text(wf["jobs"][n]) for n in offenders
    )


def test_ci_lint_job_cannot_pass_on_violation():
    job = _load()["jobs"]["lint"]
    forbidden = ("--exit-zero", "--fix", "|| true", "--diff || ", "continue-on-error")
    ruff_steps = [
        s
        for s in job["steps"]
        if "ruff check" in s.get("run", "") or "ruff format" in s.get("run", "")
    ]
    assert ruff_steps
    for step in ruff_steps:
        run = step["run"].strip()
        assert "\n" not in run, f"ruff command must be a whole step: {run!r}"
        assert run.startswith("python -m ruff "), run
        for bad in forbidden:
            assert bad not in run, f"{bad!r} in {run!r}"
        assert "continue-on-error" not in step
