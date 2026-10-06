"""Tests that enforce repository rules which are otherwise broken by accident.

These are real tests. Unlike the rest of the suite they do not test the package's
behaviour; they test the repository's own invariants, and they run in CI so that a
rule does not depend on somebody remembering it during review.
"""

from __future__ import annotations

import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The author's private planning pack. It is deliberately kept in docs/ and is
# excluded by .gitignore. A single `git add -f` would publish it, which is why
# this file exists.
PRIVATE_PACK_PATTERNS = ("docs/*.docx", "docs/**/*.docx", "*.docx")

SECRET_PATTERNS = (
    ".env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    ".netrc",
)

# Commercial inference and search providers. ADR-0001 commits the runtime to
# working without any of them, so one appearing as a dependency is an
# architectural change that needs the ADR revisited rather than a quiet addition.
FORBIDDEN_DEPENDENCIES = (
    "openai",
    "anthropic",
    "google-generativeai",
    "google-genai",
    "groq",
    "cohere",
    "mistralai",
    "replicate",
    "together",
    "ai21",
    "voyageai",
    "serpapi",
    "tavily-python",
    "youapi",
    "bing-search",
)

MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(\s*(<[^>]*>|[^)\s]+)")
FENCE = re.compile(r"^\s*(```|~~~)")

MAX_TRACKED_FILE_BYTES = 1024 * 1024


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@pytest.fixture(scope="session")
def tracked_files() -> list[Path]:
    try:
        listing = _git("ls-files", "-z")
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover
        pytest.skip("not a git working tree")
    return [REPO_ROOT / name for name in listing.split("\0") if name]


def test_private_documentation_pack_is_never_tracked(tracked_files: list[Path]) -> None:
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in tracked_files
        if path.suffix == ".docx" and path.parts[len(REPO_ROOT.parts)] == "docs"
    ]
    assert not offenders, (
        f"The private documentation pack is tracked. It must never be committed: {offenders}"
    )


def test_gitignore_declares_the_private_documentation_pack() -> None:
    # The test above only catches a leak that has already happened. This one
    # catches the rule being removed from .gitignore before it can.
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docs/*.docx" in ignored
    assert "PRIVATE WORKING MATERIAL" in ignored


def test_no_secret_shaped_files_are_tracked(tracked_files: list[Path]) -> None:
    offenders = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in tracked_files
        if any(path.match(pattern) for pattern in SECRET_PATTERNS)
    ]
    assert not offenders, f"Files that look like secrets are tracked: {offenders}"


def test_no_large_files_are_tracked(tracked_files: list[Path]) -> None:
    # Weights, dataset shards and indexes belong outside version control. See
    # docs/internal/engineering/artifact-policy.md.
    oversized = [
        (path.relative_to(REPO_ROOT).as_posix(), path.stat().st_size)
        for path in tracked_files
        if path.is_file() and path.stat().st_size > MAX_TRACKED_FILE_BYTES
    ]
    assert not oversized, f"Large files are tracked: {oversized}"


def test_no_dependency_is_a_commercial_model_or_search_provider() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    declared: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)

    requirements = [
        re.split(r"[<>=!\[; ]", item, maxsplit=1)[0].strip().lower() for item in declared
    ]
    forbidden = sorted(set(requirements) & set(FORBIDDEN_DEPENDENCIES))

    assert not forbidden, (
        "A commercial inference or search provider is declared as a dependency. "
        f"ADR-0001 forbids this: {forbidden}"
    )


def test_decision_records_are_numbered_contiguously() -> None:
    records = sorted((REPO_ROOT / "docs" / "decisions").glob("ADR-*.md"))
    numbers = []
    for record in records:
        match = re.match(r"ADR-(\d{4})-", record.name)
        assert match, f"Decision record name is not ADR-NNNN-*: {record.name}"
        numbers.append(int(match.group(1)))

    assert numbers == list(range(1, len(numbers) + 1)), (
        f"Decision record numbers have a gap or duplicate: {numbers}"
    )


def test_every_decision_record_is_indexed() -> None:
    index = (REPO_ROOT / "docs" / "decisions" / "README.md").read_text(encoding="utf-8")
    missing = [
        record.name
        for record in sorted((REPO_ROOT / "docs" / "decisions").glob("ADR-*.md"))
        if record.name not in index
    ]
    assert not missing, f"Decision records missing from the index: {missing}"


def test_markdown_relative_links_resolve(tracked_files: list[Path]) -> None:
    broken: list[str] = []

    for path in tracked_files:
        if path.suffix != ".md" or not path.is_file():
            continue
        in_fence = False
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if FENCE.match(line):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for raw_target in MARKDOWN_LINK.findall(line):
                target = raw_target.strip("<>").split("#", maxsplit=1)[0].strip()
                target = unquote(target)
                if not target or "://" in target or target.startswith(("mailto:", "tel:")):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.exists():
                    broken.append(f"{path.relative_to(REPO_ROOT).as_posix()}:{number} -> {target}")

    assert not broken, "Documentation links point at paths that do not exist:\n" + "\n".join(broken)
