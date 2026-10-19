"""Guards for dependency hygiene in pyproject.toml (issue #21)."""
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def _all_specs():
    proj = PYPROJECT["project"]
    yield from proj["dependencies"]
    for group in proj.get("optional-dependencies", {}).values():
        yield from group


def test_every_dependency_has_upper_bound():
    missing = [s for s in _all_specs() if "<" not in s and "==" not in s and "~=" not in s]
    assert not missing, f"dependencies without an upper bound: {missing}"


def test_unused_dependencies_removed():
    names = {re.split(r"[<>=~!\[ ]", s, maxsplit=1)[0].lower() for s in _all_specs()}
    assert "openai" not in names
    assert "pytest-asyncio" not in names


def test_nothing_imports_removed_dependencies():
    pat = re.compile(r"^\s*(?:import|from)\s+(openai|pytest_asyncio)\b", re.M)
    files = [*(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py"),
             *(ROOT / "scripts").rglob("*.py"), ROOT / "app.py"]
    offenders = [str(f) for f in files if f.exists() and pat.search(f.read_text())]
    assert not offenders, offenders
