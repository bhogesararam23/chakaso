"""Tests that enforce repository rules which are otherwise broken by accident.

These are real tests. Unlike the rest of the suite they do not test the package's
behaviour; they test the repository's own invariants, and they run in CI so that a
rule does not depend on somebody remembering it during review.
"""

from __future__ import annotations

import ast
import re
import subprocess
import tomllib
from pathlib import Path
from urllib.parse import unquote

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Concrete language-model adapters. ADR-0002 says application code depends on the
# boundary, never on one of these, and that configuration selects an adapter in
# exactly one place.
CONCRETE_MODEL_MODULES = frozenset(
    {
        "chakaso.models.deterministic",
        "chakaso.models.nntransformer",
        "chakaso.models.trainedchar",
    }
)

# The modules allowed to name a concrete adapter: the registry, which is the single
# composition point, and the package facade, which is where the built-in adapters
# are offered to it. `deterministic` itself is covered because it cannot import
# itself.
COMPOSITION_POINT_MODULES = frozenset(
    {
        "src/chakaso/models/__init__.py",
        "src/chakaso/models/deterministic.py",
        "src/chakaso/models/registry.py",
    }
)

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


def test_private_documentation_pack_on_disk_is_actually_ignored() -> None:
    # The tests above check that no .docx is tracked and that the ignore rule is
    # still written in .gitignore. Neither checks that the rule *works*: a typo in
    # the pattern, or an earlier negation, would pass both while leaving eight
    # private documents one `git add -A` away from being published.
    #
    # Skipped when the pack is absent, which is the case for anyone who is not the
    # project author. That is the correct behaviour, and it means this test only
    # protects the checkout where the material exists.
    present = sorted((REPO_ROOT / "docs").glob("*.docx"))
    if not present:
        pytest.skip("private documentation pack is not present in this checkout")

    ignored = _ignored_paths(present)

    assert ignored == {str(path) for path in present}, (
        "The private documentation pack is present but not excluded by .gitignore. "
        f"Not ignored: {sorted(str(path) for path in present)} - {sorted(ignored)}"
    )


def test_no_source_file_is_ignored() -> None:
    # A .gitignore pattern without a leading separator matches at any depth. The
    # first version of this repository's .gitignore contained a bare `models/` to
    # exclude model weights, which also excluded `src/chakaso/models/`: an entire
    # package, invisible to `git status`, one commit away from being silently
    # absent from the repository.
    roots = [REPO_ROOT / "src", REPO_ROOT / "tests", REPO_ROOT / "configs"]
    candidates = [
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file() and not _is_generated(path)
    ]

    ignored = _ignored_paths(candidates)

    assert not ignored, (
        "Source files are excluded by .gitignore. Anchor the offending pattern with a "
        f"leading slash: {sorted(ignored)}"
    )


def _is_generated(path: Path) -> bool:
    """Whether ``path`` is a build or tool artifact rather than source.

    Generated directories are supposed to be ignored, so their presence is not
    evidence that a source pattern went wrong. The list is explicit rather than
    "anything hidden", so that a hidden source file which became ignored is still
    reported.
    """
    generated = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
    return any(part in generated or part.endswith(".egg-info") for part in path.parts)


def _ignored_paths(paths: list[Path]) -> set[str]:
    """Return the subset of ``paths`` that git reports as ignored.

    One `git check-ignore --quiet` per path, checking the exit code, rather than a
    single `--stdin` call. The batch form looks tidier and is wrong on Windows: git
    applies C-style quoting to the paths it echoes back, so a comparison against the
    original paths silently never matches. It failed in the safe direction here
    (nothing reported as ignored), but a check that cannot fail is not a check.
    """
    ignored: set[str] = set()
    for path in paths:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            ignored.add(str(path))
    return ignored


def test_no_module_outside_the_composition_point_imports_a_concrete_model() -> None:
    # ADR-0002 is the reason a future local model is a configuration change rather
    # than a rewrite. That property is invisible in behaviour until the day someone
    # swaps the adapter, so it is checked structurally instead: a module that
    # imports a concrete adapter has taken a dependency the architecture forbids,
    # and review is not a reliable detector of one added for convenience.
    #
    # Imports are read from the syntax tree rather than the text, so a mention in a
    # docstring or comment is not a violation.
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in COMPOSITION_POINT_MODULES:
            continue
        for imported in _imported_modules(path):
            if imported in CONCRETE_MODEL_MODULES:
                offenders.append(f"{relative} -> {imported}")

    assert not offenders, (
        "A module outside the composition point imports a concrete model adapter, "
        "which ADR-0002 forbids. Import the boundary from `chakaso.models` instead, "
        f"and select an adapter through `create_model`: {offenders}"
    )


def test_the_composition_point_allowlist_does_not_go_stale() -> None:
    # An allowlist that names files which no longer exist silently stops protecting
    # anything, and the test above would keep passing.
    missing = sorted(
        relative for relative in COMPOSITION_POINT_MODULES if not (REPO_ROOT / relative).is_file()
    )

    assert not missing, f"Allowlisted modules do not exist: {missing}"


def _imported_modules(path: Path) -> set[str]:
    """Return every dotted module name ``path`` imports, absolutely."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module)
    return imported


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


def test_the_base_install_declares_no_runtime_dependency() -> None:
    # ADR-0001 keeps the runtime working with no third-party install, and ADR-0030 keeps the
    # numerical stack optional on top of that. If a dependency ever lands in `dependencies` or in
    # `dev` (which the core CI installs), the base install and CI silently stop being dependency-free.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]

    assert project.get("dependencies", []) == [], (
        "the base install must declare no runtime dependency"
    )

    dev = [
        re.split(r"[<>=!\[; ]", item, maxsplit=1)[0].strip().lower()
        for item in project["optional-dependencies"]["dev"]
    ]
    assert "numpy" not in dev, (
        "numpy belongs only in the optional [train] extra, never in dev or core CI"
    )


def test_the_training_extra_is_numpy_and_bounded_for_the_python_floor() -> None:
    # ADR-0030 admits exactly one numerical dependency, and bounds it so the declared Python floor
    # (3.11) still installs it — NumPy 2.5 dropped 3.11. This test is the enforced form of that
    # promise, so a future widening of the bound that breaks the floor is a visible decision, not a
    # silent CI failure on the oldest interpreter.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]

    assert "train" in extras, "the [train] extra named by ADR-0030 is missing"
    requirements = [item.strip().lower() for item in extras["train"]]
    assert len(requirements) == 1, f"the [train] extra must carry exactly numpy: {requirements}"
    assert requirements[0].startswith("numpy"), requirements
    # The upper bound must stay below 2.5 to keep Python 3.11 installable.
    match = re.search(r"<\s*(\d+)\.(\d+)", requirements[0])
    assert match is not None, f"the [train] numpy bound must cap the major.minor: {requirements[0]}"
    major, minor = int(match.group(1)), int(match.group(2))
    assert (major, minor) <= (2, 5), (
        f"numpy bound {requirements[0]!r} must not admit a 3.11-breaking release"
    )


# ADR-0034 admits `torch` as an optional `[nn]` extra, and admits it *only* for the modules listed here.
# The permission is a named list rather than a rule about import style because the risk is not an import
# written badly — it is a default path (any CLI command, the runtime, retrieval) silently acquiring a
# ~124 MB dependency and a ~3 s cold import. A structural check is the only thing that catches that before
# it ships.
TORCH_IMPORT_MODULES = frozenset(
    {
        "src/chakaso/training/transformer.py",
        "src/chakaso/training/decoder.py",
        "src/chakaso/training/nn_training.py",
        "src/chakaso/training/nn_evaluation.py",
        "src/chakaso/training/nn_checkpoint.py",
        "src/chakaso/training/subword_experiment.py",
        "src/chakaso/models/nntransformer.py",
    }
)


def _imports_torch(path: Path) -> bool:
    """Whether ``path`` imports ``torch`` anywhere, including inside a function body."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        else:
            continue
        if any(name == "torch" or name.startswith("torch.") for name in names):
            return True
    return False


def test_only_the_admitted_model_path_modules_import_torch() -> None:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "src").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in TORCH_IMPORT_MODULES:
            continue
        if _imports_torch(path):
            offenders.append(relative)

    assert not offenders, (
        "A module imports torch outside the allowlist ADR-0034 admits. torch is an optional [nn] "
        "dependency of the model path only: importing it anywhere else makes a ~124 MB wheel and a ~3 s "
        f"cold import load-bearing on a default path: {offenders}"
    )


def test_the_torch_allowlist_does_not_go_stale() -> None:
    # Permission for a module that never uses it is the same silent-loss-of-protection as an allowlist of
    # files that no longer exist: the next real offender looks like it is covered.
    unused = sorted(
        relative
        for relative in TORCH_IMPORT_MODULES
        if (REPO_ROOT / relative).is_file() and not _imports_torch(REPO_ROOT / relative)
    )

    assert not unused, (
        "These modules are allowed to import torch but do not; remove them from the ADR-0034 allowlist: "
        f"{unused}"
    )


def test_torch_is_declared_only_in_the_optional_nn_extra() -> None:
    # The tiers are the promise: base install dependency-free, [train] exactly numpy, torch only in [nn].
    # If torch ever lands in dependencies or dev, the dependency-free CI matrix silently stops being
    # dependency-free — which is what this test exists to prevent.
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    extras = project["optional-dependencies"]

    assert project.get("dependencies", []) == []
    for tier in ("dev", "train"):
        requirements = [
            re.split(r"[<>=!\[; ]", item, maxsplit=1)[0].strip().lower() for item in extras[tier]
        ]
        assert "torch" not in requirements, f"torch must not be declared in [{tier}]"

    assert "nn" in extras, "the [nn] extra named by ADR-0034 is missing"
    requirements = [item.strip().lower() for item in extras["nn"]]
    assert any(item.startswith("torch") for item in requirements), requirements
    torch_spec = next(item for item in requirements if item.startswith("torch"))
    assert ">=2.6" in torch_spec, (
        f"{torch_spec!r}: the 2.6 floor is where torch.load defaults to weights_only=True, which the "
        "checkpoint security rule depends on"
    )
    assert "<3" in torch_spec, f"{torch_spec!r} must bound the major version"


def test_the_two_mypy_configurations_check_the_same_strictness() -> None:
    # `mypy src` excludes the optional-dependency path because the dependency-free CI matrix cannot resolve
    # numpy or torch. That is only honest if those modules are actually checked somewhere, and a command-line
    # --exclude *adds* to the configured list instead of replacing it — so the escape hatch is a second
    # config file, and this test is what stops it drifting into a weaker one.
    default = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "mypy"
    ]
    model_path = tomllib.loads(
        (REPO_ROOT / "configs" / "mypy-model-path.toml").read_text(encoding="utf-8")
    )["tool"]["mypy"]

    for flag in (
        "strict",
        "warn_unused_ignores",
        "warn_redundant_casts",
        "show_error_codes",
        "pretty",
    ):
        assert model_path.get(flag) == default.get(flag), (
            f"{flag} differs between the configurations"
        )

    assert model_path["exclude"] == [], "the model-path configuration must not exclude anything"
    assert model_path["files"] == ["src"], model_path["files"]


def test_the_mypy_exclusion_names_only_optional_dependency_modules() -> None:
    default = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["tool"][
        "mypy"
    ]

    assert sorted(default["exclude"]) == [
        "chakaso/models/nntransformer",
        "chakaso/models/trainedchar",
        "chakaso/training",
    ], (
        "these are the modules whose types resolve through [train] or [nn]; adding one here hides it from "
        "`mypy src` silently, and removing one without removing its dependency breaks the dependency-free matrix"
    )
    for entry in default["exclude"]:
        target = REPO_ROOT / "src" / Path(*entry.split("/"))
        if target.is_dir():
            checked = list(target.rglob("*.py"))
        else:
            module = target.with_suffix(".py")
            checked = [module] if module.is_file() else []
        assert checked, f"{entry!r} excludes nothing that exists — the exclusion is stale"


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


# ---------------------------------------------------------------------------
# The optional-dependency CI matrix, as shipped


CI_MATRIX_DOCUMENT = REPO_ROOT / "docs/internal/engineering/optional-dependency-ci-matrix.md"


def _yaml_blocks(text: str) -> list[str]:
    """The fenced ```yaml blocks of a document, in order."""
    blocks: list[str] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if current is None and line.startswith("```yaml"):
            current = []
            continue
        if current is not None:
            if line.startswith("```"):
                blocks.append(chr(10).join(current))
                current = None
            else:
                current.append(line)
    return blocks


def _body(job_lines: list[str]) -> str:
    """A job's lines as one text block, joined with the newline they were split on."""
    return chr(10).join(job_lines)


def _jobs(block: str) -> dict[str, list[str]]:
    """job name -> the lines of its body, for a block whose `jobs:` mapping is indented by two.

    A hand-rolled reader rather than a YAML parser, because PyYAML is not a dependency of this project and a
    CI file is not worth adding one for: the shape it has to understand is a list of two-space-indented job
    names, and if the document ever stops matching that, these tests fail loudly instead of reading nothing.
    """
    jobs: dict[str, list[str]] = {}
    name: str | None = None
    for line in block.splitlines():
        header = re.match(r"^  ([A-Za-z][\w-]*):$", line)
        if header:
            name = header.group(1)
            jobs[name] = []
            continue
        if name is not None and line.strip():
            jobs[name].append(line)
    return jobs


CI_WORKFLOW = REPO_ROOT / ".github/workflows/ci.yml"


def _workflow_text() -> str:
    return CI_WORKFLOW.read_text(encoding="utf-8")


def _workflow_jobs() -> dict[str, list[str]]:
    """The job bodies of the shipped workflow, read from the ``jobs:`` mapping onward.

    The offset matters: the file's ``on:`` block also nests its triggers two spaces deep, and a reader that
    starts at the top would report ``push`` as a job.
    """
    lines = _workflow_text().splitlines()
    if "jobs:" not in lines:
        message = f"{CI_WORKFLOW.relative_to(REPO_ROOT)} has no top-level jobs: mapping"
        raise AssertionError(message)
    return _jobs(chr(10).join(lines[lines.index("jobs:") :]))


def test_the_shipped_workflow_installs_exactly_the_documented_tiers() -> None:
    """Every tier the package declares gets a job, and no job installs a tier it does not name.

    The workflow and pyproject are the two places this can drift, and a job that silently installs `.[dev]`
    while claiming to cover `[nn]` would make the coverage claim false without any test failing.
    """
    package = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    tiers = {
        "checks": ["dev"],
        "train": ["dev", "train"],
        "model-path": ["dev", "train", "nn"],
    }
    jobs = _workflow_jobs()
    assert jobs, "the shipped workflow carries no job bodies to read"

    assert set(jobs) == set(tiers)
    for job, extras in tiers.items():
        installs = [line.strip() for line in jobs[job] if "pip install -e" in line]
        assert len(installs) == 1, f"{job} must install exactly one set of extras"
        for extra in extras:
            assert extra in package["optional-dependencies"], f"{extra} is no longer an extra"
        declared = installs[0].split('"')[1]
        assert declared == f".[{','.join(extras)}]"


def test_the_shipped_workflow_cannot_hide_a_failure() -> None:
    block = _workflow_text()

    assert "continue-on-error" not in block
    assert "|| true" not in block
    assert "if: failure()" not in block
    for job_lines in _workflow_jobs().values():
        for line in job_lines:
            assert " --deselect" not in line and "-k " not in line, line


def test_the_shipped_workflow_runs_the_documented_commands_and_the_cpu_wheel_index() -> None:
    """A CI job that runs a weaker command than the contributor contract is not coverage of it."""
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    contributing = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    jobs = _workflow_jobs()

    for gate in ("ruff format --check .", "ruff check .", "mypy src"):
        assert f"python -m {gate}" in agents
        assert f"python -m {gate}" in contributing
    for line in jobs["checks"]:
        if line.strip().startswith("run: python -m"):
            command = line.strip().removeprefix("run: ").removeprefix("python -m ")
            assert command in agents, f"{command} is run in CI but documented nowhere"
    assert "python -m pytest" in _body(jobs["train"])
    assert "python -m mypy --config-file configs/mypy-model-path.toml" in _body(jobs["model-path"])
    model_path_runs = _body(jobs["model-path"])
    assert "https://download.pytorch.org/whl/cpu" in model_path_runs
    assert "[dev,train,nn]" in model_path_runs


def test_the_dependency_free_tier_covers_the_serving_adapter() -> None:
    """The HTTP surface has to be tested where a numerical library is absent, or its tier is a fiction.

    `checks` installs `.[dev]` and runs `python -m pytest`; the serving tests only count as coverage of the
    default install if nothing under `src/chakaso/serve/` reaches for numpy or torch. That is also the claim
    ADR-0038 makes about the transport being standard library, so it is asserted rather than remembered.
    """
    serve_files = sorted((REPO_ROOT / "src/chakaso/serve").glob("*.py"))
    assert serve_files, "the serving adapter has no modules to check"
    for path in serve_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        assert not imported & {"numpy", "torch"}, f"{path.name} imports a numerical library"
    assert "python -m pytest" in _body(_workflow_jobs()["checks"])


def test_the_ci_document_describes_the_shipped_workflow() -> None:
    """The internal document is a description of the file, not an alternative copy of it.

    It used to record a matrix that could not be pushed. If it drifts from what is shipped — naming a job
    that is not there, or still claiming the delivery is blocked while the file installs the extras — the
    next reader is told a falsehood about the project's own evidence.
    """
    document = CI_MATRIX_DOCUMENT.read_text(encoding="utf-8")
    shipped_names = set(_workflow_jobs())

    for job in sorted(shipped_names):
        assert f"`{job}`" in document, f"{job} is shipped but not described in the CI document"
    blocked = re.search(
        r"(?i)(is not running|blocked from delivery|has not run on GitHub Actions)", document
    )
    assert blocked is None, (
        f"the CI document still claims delivery is blocked: {blocked and blocked.group(0)!r} — "
        "update it with what the runners actually did"
    )
