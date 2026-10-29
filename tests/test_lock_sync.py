"""Declaration-level sync check between pyproject.toml and uv.lock (stdlib only).

Compares names, specifier sets, extras, dependency groups and requires-python.
It does not re-resolve; `uv lock --check` (local pre-land gate) keeps that job.
"""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

_REQ = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*?)\s*$")


def _norm_name(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def _norm_spec(s: str) -> frozenset[str]:
    return frozenset(re.sub(r"\s+", "", c) for c in s.split(",") if c.strip())


def _parse_req(text: str, extra: str | None = None) -> tuple[str, frozenset[str], str | None]:
    if ";" in text or "@" in text:
        raise AssertionError(
            f"requirement {text!r} uses an environment marker or URL; extend _parse_req"
        )
    m = _REQ.match(text)
    if not m:
        raise AssertionError(f"cannot parse requirement {text!r}")
    return (_norm_name(m.group(1)), _norm_spec(m.group(2)), extra)


def _lock_entry(e: dict) -> tuple[str, frozenset[str], str | None]:
    extra = None
    marker = e.get("marker")
    if marker is not None:
        m = re.fullmatch(r"\s*extra\s*==\s*'([^']+)'\s*", marker)
        if not m:
            raise AssertionError(f"unsupported marker {marker!r} on lock entry {e.get('name')}")
        extra = m.group(1)
    return (_norm_name(e["name"]), _norm_spec(e.get("specifier", "")), extra)


def _root(lock: dict) -> dict:
    found = [
        p
        for p in lock.get("package", [])
        if p.get("name") == "pixelgen" and p.get("source", {}).get("editable") == "."
    ]
    assert len(found) == 1, "no single root `pixelgen` editable package found in uv.lock"
    return found[0]


def _dist_sets(pyproject: dict, lock: dict):
    proj = pyproject["project"]
    want = {_parse_req(r) for r in proj.get("dependencies", [])}
    for extra, reqs in proj.get("optional-dependencies", {}).items():
        want |= {_parse_req(r, extra) for r in reqs}
    have = {_lock_entry(e) for e in _root(lock)["metadata"].get("requires-dist", [])}
    return want, have


def _fmt(items) -> list[str]:
    return sorted(f"{n} {','.join(sorted(s))} extra={x}" for n, s, x in items)


def _diff(label: str, want, have) -> list[str]:
    out = [f"{label}: pyproject-only {x}" for x in _fmt(want - have)]
    out += [f"{label}: lock-only {x}" for x in _fmt(have - want)]
    return out


def _compare_dist(pyproject: dict, lock: dict) -> list[str]:
    want, have = _dist_sets(pyproject, lock)
    return _diff("requires-dist", want, have)


def _compare_groups(pyproject: dict, lock: dict) -> list[str]:
    groups = pyproject.get("dependency-groups", {})
    locked = _root(lock)["metadata"].get("requires-dev", {})
    out = []
    for g in sorted(set(groups) | set(locked)):
        if g not in groups:
            out.append(f"requires-dev: lock-only group {g}")
        elif g not in locked:
            out.append(f"requires-dev: pyproject-only group {g}")
        else:
            want = {_parse_req(r) for r in groups[g]}
            have = {_lock_entry(e) for e in locked[g]}
            out += _diff(f"requires-dev[{g}]", want, have)
    return out


def _compare_python(pyproject: dict, lock: dict) -> list[str]:
    want = _norm_spec(pyproject["project"].get("requires-python", ""))
    have = _norm_spec(lock.get("requires-python", ""))
    if want != have:
        return [f"requires-python: pyproject {sorted(want)} vs lock {sorted(have)}"]
    return []


def compare(pyproject: dict, lock: dict) -> list[str]:
    """Human-readable differences; an empty list means in sync."""
    return (
        _compare_dist(pyproject, lock)
        + _compare_groups(pyproject, lock)
        + _compare_python(pyproject, lock)
    )


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    return tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))


def _assert_in_sync(diffs: list[str]) -> None:
    assert not diffs, "pyproject.toml and uv.lock differ (relock?):\n" + "\n".join(diffs)


def test_lock_root_package_is_editable_pixelgen(lock):
    root = _root(lock)
    assert root["name"] == "pixelgen"
    assert root["source"] == {"editable": "."}


def test_lock_requires_dist_matches_pyproject(pyproject, lock):
    _assert_in_sync(_compare_dist(pyproject, lock))


def test_lock_requires_dev_matches_dependency_groups(pyproject, lock):
    _assert_in_sync(_compare_groups(pyproject, lock))


def test_lock_requires_python_matches_pyproject(pyproject, lock):
    _assert_in_sync(_compare_python(pyproject, lock))


def _add_runtime(p):
    p["project"]["dependencies"].append("click>=8")


def _drop_runtime(p):
    p["project"]["dependencies"] = [
        d for d in p["project"]["dependencies"] if not d.startswith("rich")
    ]


def _bump_specifier(p):
    deps = p["project"]["dependencies"]
    deps[deps.index("rich>=13.7")] = "rich>=14"


def _add_extra_member(p):
    p["project"]["optional-dependencies"]["dev"].append("build>=1")


def _add_group_member(p):
    p["dependency-groups"]["dev"].append("build>=1")


@pytest.mark.parametrize(
    "edit",
    [_add_runtime, _drop_runtime, _bump_specifier, _add_extra_member, _add_group_member],
    ids=["add-runtime", "drop-runtime", "bump-specifier", "add-extra-member", "add-group-member"],
)
def test_comparator_sees_drift(edit, pyproject, lock):
    assert compare(pyproject, lock) == [], "base files must be in sync"
    edited = copy.deepcopy(pyproject)
    edit(edited)
    assert compare(edited, lock), f"{edit.__name__} was not detected"


def test_parser_rejects_marker_or_url_requirement():
    with pytest.raises(AssertionError, match="foo"):
        _parse_req('foo>=1; python_version < "3.12"')
    with pytest.raises(AssertionError, match="foo"):
        _parse_req("foo @ https://x/y.whl")
