"""Command line entry point.

The CLI is deliberately thin. It exists so that the package can be verified after
installation (``chakaso --version``), so that configuration can be inspected
(``chakaso config show``), so that the conversation boundary can be exercised by a
person (``chakaso chat``), and so that retrieval can be run over named local files
(``chakaso retrieve``). It does not implement behaviour of its own: composing a model,
a corpus, a retrieval service or a manager is wiring, and wiring belongs at the entry
point. Every command calls the same application layer the library exposes.

``chat`` is honest about what answers it. The only adapter that exists is a
deterministic development double, and the command says so in its own output rather
than letting a user infer that the replies mean something. ``retrieve`` never generates
an answer; it ingests, ranks and prints evidence with its provenance. ``benchmark``
runs a retrieval, correction, or strategy-comparison development benchmark over the shared
synthetic fixtures and prints a report that labels itself a development instrument
(``benchmark strategies`` measures lexical, dense and hybrid over the same data and reports the
measured deltas); ``experiment run`` executes a
baseline experiment and prints its reproducible, content-fingerprinted result.
``storage``
checks or migrates a durable SQLite answer store and reports its schema and integrity, reading
or writing only the explicit path it is given.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from chakaso import __version__
from chakaso.answers import AnswerError, AnswerId, AnswerRecord, SQLiteAnswerStore
from chakaso.app import (
    Application,
    ApplicationError,
    ApplicationTurn,
    Capabilities,
    build_application,
    build_corpus,
    describe_capabilities,
)
from chakaso.benchmark import (
    BenchmarkError,
    CorrectionBenchmarkRunner,
    RetrievalBenchmarkRunner,
    correction_report_json,
    development_correction_benchmark,
    format_correction_report,
    format_report,
    load_development_benchmark,
    report_json,
)
from chakaso.config import ConfigError, LoadedConfig, RetrievalConfig, load_config
from chakaso.conversation import Reply
from chakaso.core.errors import ChakasoError
from chakaso.evaluation.semantic_benchmark import (
    ProbeSupportAllJudge,
    ProbeUncertainJudge,
    development_semantic_dataset,
    format_semantic_report,
    materialize_case,
    score_semantic_judge,
)
from chakaso.experiments import (
    BenchmarkKind,
    RetrievalStrategyReport,
    baseline_experiment,
    compare_retrieval_strategies,
    run_experiment,
)
from chakaso.grounding import FixtureSemanticJudge
from chakaso.models import UnknownModelError
from chakaso.planning import QueryMode
from chakaso.retrieval import (
    DEFAULT_TOP_K,
    Corpus,
    IngestionError,
    RetrievalError,
    RetrievalOutcome,
    RetrievalService,
    ingest_file,
)
from chakaso.runtime import build_retrieval_services

__all__ = ["build_parser", "main"]

PROGRAM = "chakaso"

#: Interactive commands that end the session. Prefixed with a colon so that they
#: cannot be confused with something a user wants to ask.
_QUIT_COMMANDS = frozenset({":quit", ":exit"})


def build_parser() -> argparse.ArgumentParser:
    """Build the root argument parser.

    Kept separate from :func:`main` so that tests can assert on the parser's
    behaviour without going through the process exit path.
    """
    root, _config_parser = _build_parsers()
    return root


def _build_parsers() -> tuple[argparse.ArgumentParser, argparse.ArgumentParser]:
    """Build the root parser and the ``config`` sub-parser.

    Both are returned because ``chakaso config`` with no action prints the config
    parser's own help. Asking argparse for it by re-parsing ``["config", "--help"]``
    would raise ``SystemExit`` from inside a function that otherwise returns an exit
    code, which is a worse contract for callers and for tests.
    """
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description=(
            "Chakaso is a research project. There is a conversation shell, but no "
            "language model yet: replies come from a deterministic development double. "
            "See docs/agent/CURRENT_STATE.md in the repository."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROGRAM} {__version__}",
        help="print the installed version and exit",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.add_parser(
        "info",
        help="print runtime facts useful in a bug report",
        description=(
            "Print the package version, the Python version and the platform. "
            "This is the information a bug report needs first."
        ),
    )

    config_parser = subparsers.add_parser(
        "config",
        help="inspect configuration",
        description="Inspect the resolved configuration and where each value came from.",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", metavar="ACTION")
    show_parser = config_subparsers.add_parser(
        "show",
        help="print the resolved configuration with its provenance",
        description=(
            "Print every configuration value together with the source it came from. "
            "Nothing is loaded implicitly: pass the files to apply, in order."
        ),
    )
    _add_config_arguments(show_parser)

    chat_parser = subparsers.add_parser(
        "chat",
        help="hold a conversation through the retrieval-aware runtime",
        description=(
            "Hold a conversation grounded by the retrieval-aware runtime: the planner decides "
            "whether a turn retrieves, retrieval (lexical, dense or hybrid) runs over the "
            "documents given by --file, and each answer is recorded in the configured store. "
            "Replies still come from the model in configuration, which today is a deterministic "
            "development double: fixed text, not answers. dense/hybrid run over a non-semantic "
            "fixture embedding. Local and offline; with no --file the corpus is empty and "
            "retrieval honestly returns no evidence."
        ),
    )
    _add_config_arguments(chat_parser)
    chat_parser.add_argument(
        "--message",
        default=None,
        metavar="TEXT",
        help=(
            "send one message, print the reply and exit, instead of reading messages "
            "from standard input"
        ),
    )
    chat_parser.add_argument(
        "--file",
        action="append",
        default=None,
        metavar="PATH",
        help="a local document to add to the retrieval corpus; repeat for multiple sources",
    )
    chat_parser.add_argument(
        "--strategy",
        choices=[QueryMode.LEXICAL.value, QueryMode.DENSE.value, QueryMode.HYBRID.value],
        default=None,
        help="override the configured retrieval strategy (dense/hybrid use the fixture embedding)",
    )
    chat_parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="N",
        help="override how many chunks a retrieval turn returns",
    )
    chat_parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="run the conversation without grounding (the planner and retrieval are skipped)",
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="search supplied local documents for a query and print ranked evidence",
        description=(
            "Ingest the files given by --file into an in-memory corpus, run lexical BM25 "
            "retrieval for --query, and print the ranked evidence with its provenance. "
            "Local and offline: no network, no model, and no answer is generated."
        ),
    )
    retrieve_parser.add_argument(
        "--query",
        required=True,
        metavar="TEXT",
        help="the query to retrieve for",
    )
    retrieve_parser.add_argument(
        "--file",
        action="append",
        default=None,
        metavar="PATH",
        help="a local document to ingest; repeat for multiple sources",
    )
    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help="how many chunks to return (default: %(default)s)",
    )

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="run a development benchmark (retrieval or correction) and print its report",
        description=(
            "Run a versioned development benchmark over the shared synthetic fixtures and print "
            "a report. Both subjects are development instruments: they score a component against a "
            "small hand-built set, run no language model, and cannot support any claim about "
            "general performance. Local, offline and deterministic."
        ),
    )
    benchmark_subparsers = benchmark_parser.add_subparsers(
        dest="benchmark_command", metavar="SUBJECT"
    )

    retrieval_benchmark = benchmark_subparsers.add_parser(
        "retrieval",
        help="score the lexical retriever against the retrieval development benchmark",
        description=(
            "Run the retrieval development benchmark (chakaso.benchmark) over its synthetic "
            "fixture corpus. It scores retrieval only and evaluates no answer."
        ),
    )
    retrieval_benchmark.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help="how many chunks to retrieve per case (default: %(default)s)",
    )
    retrieval_benchmark.add_argument(
        "--strategy",
        choices=["lexical", "dense", "hybrid"],
        default="lexical",
        help="which retriever to score (default: %(default)s; dense/hybrid use the fixture embedding)",
    )
    retrieval_benchmark.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic machine-readable report instead of the text one",
    )

    correction_benchmark = benchmark_subparsers.add_parser(
        "correction",
        help="score the correction decision rule against the correction development benchmark",
        description=(
            "Run the correction development benchmark (chakaso.benchmark.correction): synthetic "
            "cases whose expected decision the correction rule is checked against. It scores a "
            "decision rule over fixtures, runs no model, and infers no contradiction."
        ),
    )
    correction_benchmark.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic machine-readable report instead of the text one",
    )

    strategies_benchmark = benchmark_subparsers.add_parser(
        "strategies",
        help="measure and compare lexical, dense and hybrid retrieval over the development fixtures",
        description=(
            "Run the retrieval development benchmark with each retriever over the same synthetic "
            "dataset and report the measured metrics and deltas against a baseline strategy. Dense and "
            "hybrid use the non-semantic fixture embedding, so results describe mechanism, not real or "
            "semantic quality. Local, offline, deterministic; no winner is hard-coded."
        ),
    )
    strategies_benchmark.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="N",
        help="how many chunks to retrieve per case for every strategy (default: %(default)s)",
    )
    strategies_benchmark.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic machine-readable report instead of the text one",
    )

    semantic_benchmark = benchmark_subparsers.add_parser(
        "semantic",
        help="score a semantic judge over the synthetic semantic-evaluation dataset",
        description=(
            "Run the semantic-evaluation development instrument: a versioned, hand-labelled synthetic "
            "set of claim/evidence cases scored against the only judges available today — a fixture and "
            "two named probes — reporting accuracy, per-class precision/recall, a confusion matrix, "
            "unresolved and error handling. It measures the SCORING HARNESS, not semantic understanding: "
            "no real evaluator exists (ADR-0019/ADR-0027), so no number here is general language ability. "
            "Local, offline, deterministic."
        ),
    )
    semantic_benchmark.add_argument(
        "--json",
        action="store_true",
        help="emit the deterministic machine-readable report instead of the text one",
    )

    experiment_parser = subparsers.add_parser(
        "experiment",
        help="run a development experiment and print its reproducible result",
        description=(
            "Run a canonical development experiment against its benchmark and print the "
            "reproducible, content-fingerprinted result. Development measurements over synthetic "
            "fixtures; no model runs, and no real-world performance is claimed."
        ),
    )
    experiment_subparsers = experiment_parser.add_subparsers(
        dest="experiment_command", metavar="ACTION"
    )
    experiment_run = experiment_subparsers.add_parser(
        "run",
        help="run a baseline experiment (retrieval or correction)",
        description="Run a baseline development experiment and print its result.",
    )
    experiment_run.add_argument(
        "subject",
        choices=[kind.value for kind in BenchmarkKind],
        help="which baseline experiment to run",
    )
    experiment_run.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable result record instead of a one-line summary",
    )

    storage_parser = subparsers.add_parser(
        "storage",
        help="check or migrate a durable answer store",
        description=(
            "Inspect or bring forward a durable SQLite answer store (ADR-0021). Local and offline: "
            "it opens the file named by --path (or by configuration) and reports or migrates its "
            "schema. Nothing is created or changed unless a real path is given; the in-memory default "
            "backend has nothing durable to inspect."
        ),
    )
    storage_subparsers = storage_parser.add_subparsers(dest="storage_command", metavar="ACTION")
    storage_check = storage_subparsers.add_parser(
        "check",
        help="report a durable store's schema version and integrity",
        description="Read a durable store's structure without writing to it.",
    )
    _add_storage_arguments(storage_check)
    storage_migrate = storage_subparsers.add_parser(
        "migrate",
        help="initialize or bring a durable store's schema up to date",
        description="Create a fresh store schema, or confirm an existing one is current.",
    )
    _add_storage_arguments(storage_migrate)

    capabilities_parser = subparsers.add_parser(
        "capabilities",
        help="report the model, retrieval and storage the configuration selects",
        description=(
            "Print what Chakaso would wire from the resolved configuration: the model adapter and "
            "whether it is a development double, the retrieval strategy and limits, and the answer-"
            "store backend. Local, offline and read-only — it opens no store and reads no files."
        ),
    )
    _add_config_arguments(capabilities_parser)
    capabilities_parser.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable capability report instead of a text summary",
    )

    answers_parser = subparsers.add_parser(
        "answers",
        help="inspect persisted answers and their correction lineage",
        description=(
            "Read recorded answers from the configured durable store (ADR-0021): list them, show one "
            "by answer identifier, or walk a correction chain. The in-memory default backend persists "
            "nothing across processes, so there is nothing to inspect unless [storage] selects 'sqlite' "
            "with a real file path. Local, offline and read-only."
        ),
    )
    answers_subparsers = answers_parser.add_subparsers(dest="answers_command", metavar="ACTION")
    answers_list = answers_subparsers.add_parser(
        "list", help="list recorded answers with their provenance"
    )
    _add_config_arguments(answers_list)
    answers_list.add_argument(
        "--conversation-id",
        default=None,
        metavar="ID",
        help="restrict the listing to one conversation",
    )
    answers_list.add_argument("--json", action="store_true", help="machine-readable output")
    answers_show = answers_subparsers.add_parser("show", help="show one recorded answer")
    answers_show.add_argument("answer_id", metavar="ANSWER_ID", help="the ans_… identifier to show")
    _add_config_arguments(answers_show)
    answers_show.add_argument("--json", action="store_true", help="machine-readable output")
    answers_lineage = answers_subparsers.add_parser(
        "lineage", help="walk an answer's correction chain, oldest first"
    )
    answers_lineage.add_argument(
        "answer_id", metavar="ANSWER_ID", help="the ans_… identifier whose chain to walk"
    )
    _add_config_arguments(answers_lineage)
    answers_lineage.add_argument("--json", action="store_true", help="machine-readable output")

    return parser, config_parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        default=None,
        metavar="PATH",
        help=(
            "configuration file to apply; repeat to layer files, later ones winning. "
            "Omit to use the built-in defaults."
        ),
    )


def _add_storage_arguments(parser: argparse.ArgumentParser) -> None:
    _add_config_arguments(parser)
    parser.add_argument(
        "--path",
        default=None,
        metavar="PATH",
        help="the durable store file to inspect; overrides the configured storage.path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the machine-readable report instead of a text summary",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    parser, config_parser = _build_parsers()
    args = parser.parse_args(argv)

    if args.command is None:
        # No command is not an error; the user most likely wants to know what the
        # tool does. Argparse is asked for the same help text a failure would show.
        parser.print_help()
        return 0

    if args.command == "info":
        _print_info()
        return 0

    if args.command == "config":
        return _run_config(parser, config_parser, args)

    if args.command == "chat":
        return _run_chat(args)

    if args.command == "retrieve":
        return _run_retrieve(args)

    if args.command == "benchmark":
        return _run_benchmark(args)

    if args.command == "experiment":
        return _run_experiment_command(args)

    if args.command == "storage":
        return _run_storage(args)

    if args.command == "capabilities":
        return _run_capabilities(args)

    if args.command == "answers":
        return _run_answers(args)

    # Unreachable while every subcommand is handled above. argparse rejects
    # unknown commands before this point, so reaching here means a subcommand was
    # added to the parser and not to this dispatch.
    parser.error(f"unhandled command: {args.command!r}")
    return 2


def _run_config(
    parser: argparse.ArgumentParser,
    config_parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> int:
    if args.config_command is None:
        config_parser.print_help()
        return 0

    if args.config_command == "show":
        return _show_config(args.config)

    parser.error(f"unhandled config action: {args.config_command!r}")
    return 2


def _show_config(paths: list[str] | None) -> int:
    try:
        loaded = _load_or_report(paths)
    except ConfigError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    if loaded is None:  # pragma: no cover - _load_or_report returns or raises
        return 1

    print(_format_config(loaded))
    return 0


def _load_or_report(paths: list[str] | None) -> LoadedConfig:
    """Load configuration from ``paths``, later files winning.

    Raises:
        ConfigError: the files cannot be read or do not validate.
    """
    files = [Path(item) for item in paths] if paths else []
    primary = files[0] if files else None
    return load_config(primary, overrides=files[1:])


def _apply_chat_overrides(loaded: LoadedConfig, args: argparse.Namespace) -> LoadedConfig:
    """Fold the ``chat`` command's flags into the resolved configuration.

    The typed configuration stays the single source of truth for construction; a CLI flag is an
    explicit, named override of one field, applied with :func:`dataclasses.replace` rather than a
    second configuration path. Which fields were overridden is reported separately so provenance
    is not silently rewritten to claim the file supplied them.
    """
    retrieval = loaded.config.retrieval
    enabled = False if args.no_retrieval else retrieval.enabled
    strategy = args.strategy if args.strategy is not None else retrieval.strategy
    top_k = args.top_k if args.top_k is not None else retrieval.top_k
    if (enabled, strategy, top_k) == (retrieval.enabled, retrieval.strategy, retrieval.top_k):
        return loaded
    overridden = RetrievalConfig(
        enabled=enabled,
        strategy=strategy,
        top_k=top_k,
        candidate_k=retrieval.candidate_k,
        rrf_k=retrieval.rrf_k,
        max_ingest_bytes=retrieval.max_ingest_bytes,
    )
    return replace(loaded, config=replace(loaded.config, retrieval=overridden))


def _run_chat(args: argparse.Namespace) -> int:
    if args.top_k is not None and args.top_k < 1:
        print(f"{PROGRAM}: --top-k must be at least 1", file=sys.stderr)
        return 1

    try:
        loaded = _load_or_report(args.config)
    except ConfigError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    loaded = _apply_chat_overrides(loaded, args)

    # The corpus is built from only the files the caller names, before any turn runs: reading one
    # is an explicit request, and a path that cannot be read is reported whole rather than
    # swallowed into a silently smaller corpus.
    try:
        corpus = build_corpus(
            args.file or [],
            max_bytes=loaded.config.retrieval.max_ingest_bytes,
        )
    except IngestionError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    try:
        app = build_application(loaded, corpus=corpus)
    except (UnknownModelError, ApplicationError, ChakasoError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    # Everything a person needs in order not to mistake this for a real assistant goes to
    # stderr, so that stdout carries the reply alone and the command stays scriptable.
    print(_engine_notice(app), file=sys.stderr)
    _report_config_overrides(args)

    with app:
        if args.message is not None:
            return _answer_once(app, args.message)
        return _converse(app)


def _engine_notice(app: Application) -> str:
    """State plainly what produces the replies and what grounds them, before any reply."""
    metadata = app.model_metadata
    lines = [f"engine: {metadata.model_id} ({metadata.display_name})"]
    if metadata.development_double:
        lines.append(
            "This is a development double, not a language model. It returns fixed text "
            "that exercises the conversation plumbing, and it does not answer anything."
        )
    if app.retrieval_enabled:
        corpus = app.corpus
        if corpus.is_empty:
            lines.append(
                f"retrieval: {app.strategy.value} selected, but no documents were supplied, so a "
                "retrieving turn returns no evidence (pass --file to ground answers)."
            )
        else:
            lines.append(
                f"retrieval: {app.strategy.value} over {corpus.source_count} source(s) and "
                f"{corpus.chunk_count} chunk(s). This grounds and cites structurally only — no "
                "semantic judgement, and dense/hybrid run over a non-semantic fixture embedding."
            )
    else:
        lines.append("retrieval disabled: no planner runs and turns are not grounded in evidence.")
    if app.storage_backend == "sqlite":
        lines.append(f"answers persist durably to {app.storage_path} and survive a restart.")
    else:
        lines.append(
            "answers are recorded in memory for this process only and are not saved to disk."
        )
    return "\n".join(lines)


def _report_config_overrides(args: argparse.Namespace) -> None:
    """Name the configuration the command line overrode, so a run is not mistaken for default."""
    overridden: list[str] = []
    if args.no_retrieval:
        overridden.append("retrieval.enabled=false (--no-retrieval)")
    if args.strategy is not None:
        overridden.append(f"retrieval.strategy={args.strategy!r} (--strategy)")
    if args.top_k is not None:
        overridden.append(f"retrieval.top_k={args.top_k} (--top-k)")
    if overridden:
        print(f"{PROGRAM}: overridden from configuration: {', '.join(overridden)}", file=sys.stderr)


def _answer_once(app: Application, message: str) -> int:
    try:
        turn = app.turn(message)
    except ChakasoError as exc:
        # Every deliberate failure keeps its own message; a failed turn leaves the conversation
        # unchanged, so nothing half-recorded is left behind.
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(turn.text)
    _report_caveats(turn.reply)
    _report_grounding(turn)
    return 0


def _converse(app: Application) -> int:
    """Read messages until end of input or a quit command."""
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            if text in _QUIT_COMMANDS:
                break

            try:
                turn = app.turn(text)
            except ChakasoError as exc:
                # A failed turn leaves the conversation unchanged, so the session can
                # continue rather than losing everything said so far.
                print(f"{PROGRAM}: {exc}", file=sys.stderr)
                continue

            print(f"{PROGRAM}> {turn.text}")
            _report_caveats(turn.reply)
            _report_grounding(turn)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    return 0


def _report_grounding(turn: ApplicationTurn) -> None:
    """Report what grounding actually happened on a turn — to stderr, never inventing a claim.

    This is the runtime's provenance made visible: the planner's decision and why it fired, how
    much evidence retrieval offered, and how many sources the answer actually cited. It states
    plainly that a resolved citation is structure, not a source proving a claim.
    """
    plan = turn.plan
    if plan is None:
        return

    if not turn.retrieved:
        reason = "; ".join(plan.explanation) if plan.explanation else "no reasons recorded"
        print(
            f"{PROGRAM}: no retrieval this turn (planner chose {plan.mode.value}): {reason}",
            file=sys.stderr,
        )
        return

    offered = max(len(turn.evidence_context) - 1, 0)
    cited = len(turn.reply.citation_resolution.cited_source_ids)
    print(
        f"{PROGRAM}: retrieved via {plan.mode.value}; {offered} chunk(s) offered, {cited} "
        f"source(s) cited. A resolved citation shows a source was referenced, not that it "
        "proves the answer.",
        file=sys.stderr,
    )


def _report_caveats(reply: Reply) -> None:
    """Report conditions a reader would otherwise not know about."""
    if reply.is_truncated:
        print(
            f"{PROGRAM}: the reply stopped at the length ceiling and is incomplete",
            file=sys.stderr,
        )
    if reply.citation_resolution.unknown_identifiers:
        unresolved = ", ".join(reply.citation_resolution.unknown_identifiers)
        print(
            f"{PROGRAM}: the reply referenced evidence it was not given, which was not "
            f"resolved: {unresolved}",
            file=sys.stderr,
        )
    if reply.citation_resolution.malformed_identifiers:
        malformed = ", ".join(reply.citation_resolution.malformed_identifiers)
        print(
            f"{PROGRAM}: the reply contained malformed evidence references: {malformed}",
            file=sys.stderr,
        )


def _run_retrieve(args: argparse.Namespace) -> int:
    """Ingest the given files, retrieve for the query, and print ranked evidence.

    This calls the same application layer the library exposes — ``ingest_file``,
    ``Corpus`` and ``RetrievalService`` — and adds no retrieval logic of its own.
    """
    paths = args.file or []
    if not paths:
        print(f"{PROGRAM}: retrieve needs at least one --file", file=sys.stderr)
        return 1

    try:
        corpus = Corpus()
        for path in paths:
            corpus.add_document(ingest_file(path))
        outcome = RetrievalService(corpus).search(args.query, top_k=args.top_k)
    except (IngestionError, RetrievalError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(_format_retrieval(outcome))
    return 0


def _format_retrieval(outcome: RetrievalOutcome) -> str:
    """Render a retrieval outcome for a developer, including provenance.

    The score is a ranking artefact, not a confidence, and it is labelled as such; the
    matched terms are exactly what the retriever computed, nothing more.
    """
    config = outcome.retrieval_config
    lines = [
        f"retriever: {config['retriever']}  top_k: {config['top_k']}  query: {outcome.query!r}"
    ]
    if not outcome.results:
        lines.append("no evidence matched the query")
        return "\n".join(lines)

    for result in outcome.results:
        chunk = result.chunk
        source = outcome.pack.source_for(chunk.source_id)
        title = source.title if source is not None else "?"
        reference = source.canonical_reference if source is not None else "?"
        lines.append(
            f"{result.rank}. score={result.score:.4f}  section={chunk.section or '-'}  "
            f"chunk_id={chunk.chunk_id}"
        )
        lines.append(f"   source: {title} <{reference}>")
        lines.append(f"   matched: {', '.join(result.matched_terms)}")
        lines.append(f"   {chunk.text}")
    return "\n".join(lines)


def _run_benchmark(args: argparse.Namespace) -> int:
    """Dispatch a benchmark subject. No scoring lives here — the runners and reports do."""
    if args.benchmark_command == "retrieval":
        return _run_retrieval_benchmark(args)
    if args.benchmark_command == "correction":
        return _run_correction_benchmark(args)
    if args.benchmark_command == "strategies":
        return _run_strategy_benchmark(args)
    if args.benchmark_command == "semantic":
        return _run_semantic_benchmark(args)
    print(
        f"{PROGRAM}: choose a benchmark subject: 'retrieval', 'correction', 'strategies' or 'semantic'",
        file=sys.stderr,
    )
    return 1


def _run_retrieval_benchmark(args: argparse.Namespace) -> int:
    """Run the retrieval development benchmark and print its report.

    Wires ``load_development_benchmark``, ``RetrievalService`` and ``RetrievalBenchmarkRunner``
    and adds no scoring logic of its own; the report's wording and its "development benchmark"
    label live in ``chakaso.benchmark.report``. The run is deterministic and offline.
    """
    try:
        corpus, dataset = load_development_benchmark()
        service = build_retrieval_services(corpus)[QueryMode(args.strategy)]
        runner = RetrievalBenchmarkRunner(service, top_k=args.top_k, retriever_name=args.strategy)
        run = runner.run(dataset)
    except (BenchmarkError, IngestionError, RetrievalError, ValueError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(report_json(run) if args.json else format_report(run))
    return 0


def _run_strategy_benchmark(args: argparse.Namespace) -> int:
    """Score every retrieval strategy over the same fixtures and print the measured comparison.

    Thin wiring: :func:`chakaso.experiments.compare_retrieval_strategies` runs each retriever and
    computes the deltas; this only formats its report. The numbers come from an actual run — nothing
    is hard-coded, and a strategy that does not win is reported as not winning.
    """
    try:
        corpus, dataset = load_development_benchmark()
        report = compare_retrieval_strategies(corpus, dataset, top_k=args.top_k)
    except (BenchmarkError, IngestionError, RetrievalError, ValueError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report.to_dict(), sort_keys=True, indent=2, ensure_ascii=False))
    else:
        print(_format_strategy_report(report))
    return 0


def _format_strategy_report(report: RetrievalStrategyReport) -> str:
    """Render a strategy comparison as a fixed-width table plus baseline deltas."""
    lines = [
        "retrieval strategy comparison - a development benchmark over synthetic fixtures; dense and",
        "hybrid use a non-semantic fixture embedding (not a real/semantic result). "
        f"baseline={report.baseline!r} top_k={report.top_k}",
        "",
        f"{'strategy':<9}{'recall@k':>9}{'prec@k':>9}{'mrr':>9}{'hit':>9}{'forb':>6}{'false':>7}{'dup':>5}",
    ]
    for row in report.results:
        lines.append(
            f"{row.strategy:<9}{row.recall_at_k:>9.3f}{row.precision_at_k:>9.3f}{row.mrr:>9.3f}"
            f"{row.hit_rate:>9.3f}{row.forbidden_hits:>6d}{row.false_retrievals:>7d}{row.duplicates:>5d}"
        )
    lines.append("")
    lines.append(f"deltas vs {report.baseline!r}:")
    if not report.deltas:
        lines.append("  (no non-baseline strategies were run)")
    for delta in report.deltas:
        lines.append(
            f"  {delta.strategy} {delta.metric}: {delta.baseline:.3f} -> {delta.candidate:.3f} "
            f"({delta.delta:+.3f}, {delta.direction})"
        )
    return "\n".join(lines)


def _run_correction_benchmark(args: argparse.Namespace) -> int:
    """Run the correction development benchmark and print its report.

    Wires ``CorrectionBenchmarkRunner`` over the curated dataset; the decision rule and the report
    live in ``chakaso.correction`` and ``chakaso.benchmark``. It scores a rule over synthetic
    cases — no model runs and no contradiction is inferred.
    """
    try:
        run = CorrectionBenchmarkRunner().run(development_correction_benchmark())
    except (BenchmarkError, LookupError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(correction_report_json(run) if args.json else format_correction_report(run))
    return 0


def _run_semantic_benchmark(args: argparse.Namespace) -> int:
    """Score the available judges over the semantic-evaluation development dataset.

    It runs the fixture (given the expected relations — this validates the scorer's arithmetic and
    proves nothing about understanding) and two named probes (support-all, always-uncertain) that
    show the scorer surfaces a judge that over-claims versus one that abstains. No real evaluator
    exists; every number is labelled as a development measurement of the harness.
    """
    try:
        cases = development_semantic_dataset()
        fixture = FixtureSemanticJudge(
            {materialize_case(case)[0].claim_id: case.expected for case in cases}
        )
        runs = [
            score_semantic_judge(fixture, cases=cases),
            score_semantic_judge(ProbeSupportAllJudge(), cases=cases),
            score_semantic_judge(ProbeUncertainJudge(), cases=cases),
        ]
    except (ValueError, ChakasoError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                [run.to_dict() for run in runs], sort_keys=True, indent=2, ensure_ascii=False
            )
        )
    else:
        rendered = "\n\n".join(format_semantic_report(run) for run in runs)
        print(rendered)
        print(
            "\nThe fixture's perfect score measures the scorer, not a capability: it was handed the "
            "expected relations. The probes show the scorer distinguishes over-claiming from abstention."
        )
    return 0


def _run_experiment_command(args: argparse.Namespace) -> int:
    """Run a baseline development experiment and print its reproducible result.

    Thin wiring: the experiment data, the runner and the result all live in ``chakaso.experiments``;
    this only picks the baseline by subject and formats the library's ``to_dict``. Output is
    deterministic and offline; the result's ``result_id`` is the reproducibility fingerprint.
    """
    if args.experiment_command != "run":
        print(
            f"{PROGRAM}: choose an experiment action: 'run retrieval' or 'run correction'",
            file=sys.stderr,
        )
        return 1

    try:
        result = run_experiment(baseline_experiment(BenchmarkKind(args.subject)))
    except (BenchmarkError, ValueError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result.to_dict(), sort_keys=True, indent=2, ensure_ascii=False))
    else:
        print(
            f"experiment {result.experiment_id}: result {result.result_id} "
            f"on {result.dataset_id} v{result.dataset_version}"
        )
    return 0


def _run_storage(args: argparse.Namespace) -> int:
    """Dispatch a storage action. No store logic lives here — ``check()``/``migrate()`` do."""
    action = args.storage_command
    if action not in ("check", "migrate"):
        print(f"{PROGRAM}: choose a storage action: 'check' or 'migrate'", file=sys.stderr)
        return 1

    try:
        loaded = _load_or_report(args.config)
    except ConfigError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    path = (args.path or loaded.config.storage.path).strip()
    if not path:
        print(
            f"{PROGRAM}: no durable store path is configured (backend "
            f"{loaded.config.storage.backend!r}); nothing to {action}"
        )
        return 0

    if action == "check":
        return _check_store(Path(path), args.json)
    return _migrate_store(Path(path), loaded.config.storage.journal_mode, args.json)


def _check_store(target: Path, as_json: bool) -> int:
    """Report a durable store's health without writing to it."""
    if not target.is_file():
        print(f"{PROGRAM}: no durable store file at {target}; nothing to check")
        return 0
    try:
        # initialize=False so an absent schema is reported rather than created, and
        # journal_mode=None so a read is never silently a write to the file header.
        with SQLiteAnswerStore(target, journal_mode=None, initialize=False) as store:
            report = dict(store.check())
    except (AnswerError, OSError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1
    print(_format_storage_report(report, as_json))
    return 0 if report.get("ok") else 1


def _migrate_store(target: Path, journal_mode: str, as_json: bool) -> int:
    """Create or confirm a durable store's schema, then report the outcome."""
    try:
        with SQLiteAnswerStore(target, journal_mode=journal_mode, initialize=True) as store:
            result: dict[str, object] = {"path": str(target), **store.migrate()}
    except (AnswerError, OSError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1
    print(_format_storage_report(result, as_json))
    return 0


def _format_storage_report(report: dict[str, object], as_json: bool) -> str:
    if as_json:
        return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False)
    lines: list[str] = []
    for key in sorted(report):
        value = report[key]
        rendered = value if isinstance(value, str) else json.dumps(value, sort_keys=True)
        lines.append(f"{key}: {rendered}")
    return "\n".join(lines)


def _run_capabilities(args: argparse.Namespace) -> int:
    """Report the capabilities the configuration selects, without running a turn or opening a store."""
    try:
        loaded = _load_or_report(args.config)
    except ConfigError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    try:
        capabilities = describe_capabilities(loaded)
    except ChakasoError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(capabilities.to_dict(), sort_keys=True, indent=2, ensure_ascii=False))
    else:
        print(_format_capabilities(capabilities))
    return 0


def _format_capabilities(capabilities: Capabilities) -> str:
    double = (
        "yes — returns fixed text, not answers"
        if capabilities.model_is_development_double
        else "no"
    )
    if capabilities.retrieval_enabled:
        retrieval = (
            f"on (strategy={capabilities.retrieval_strategy}, top_k={capabilities.retrieval_top_k})"
        )
    else:
        retrieval = "off"
    storage = capabilities.storage_backend
    if capabilities.storage_backend == "sqlite" and capabilities.storage_path.strip():
        storage = f"sqlite at {capabilities.storage_path}"
    if capabilities.semantic_real_evaluator_available:
        semantic = (
            f"available (judges: {', '.join(capabilities.semantic_available_judges) or 'none'})"
        )
    else:
        semantic = (
            "no real evaluator — only a caller-supplied fixture judge exists; grounding is "
            "structural"
        )
    return "\n".join(
        [
            f"model: {capabilities.model_id} ({capabilities.model_display_name})",
            f"development double: {double}",
            f"retrieval: {retrieval}",
            f"storage: {storage}",
            f"semantic evaluation: {semantic}",
        ]
    )


def _open_inspectable_store(loaded: LoadedConfig) -> tuple[SQLiteAnswerStore | None, str]:
    """Return a read-only handle on the configured durable store, or why there is nothing to read."""
    storage = loaded.config.storage
    if storage.backend != "sqlite":
        return (
            None,
            "the in-memory backend holds nothing after this process exits; configure "
            "[storage] backend='sqlite' with a path to inspect persisted answers",
        )
    path = storage.path.strip()
    if not path:
        return None, "storage.backend is 'sqlite' but storage.path is empty; nothing to inspect"
    target = Path(path)
    if not target.is_file():
        return None, f"no durable store file at {target}; nothing to inspect"
    # initialize=False and journal_mode=None so inspection never creates a database or writes to
    # the file header: reading the answer history must not mutate it.
    return SQLiteAnswerStore(target, journal_mode=None, initialize=False), ""


def _run_answers(args: argparse.Namespace) -> int:
    action = args.answers_command
    if action not in ("list", "show", "lineage"):
        print(f"{PROGRAM}: choose an answers action: 'list', 'show' or 'lineage'", file=sys.stderr)
        return 1

    try:
        loaded = _load_or_report(args.config)
    except ConfigError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    store, note = _open_inspectable_store(loaded)
    if store is None:
        print(f"{PROGRAM}: {note}")
        return 0

    try:
        with store:
            if action == "list":
                return _answers_list(store, args)
            if action == "show":
                return _answers_show(store, args)
            return _answers_lineage(store, args)
    except (AnswerError, ChakasoError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1


def _answer_brief(record: AnswerRecord) -> dict[str, object]:
    """A stable, machine-readable summary of one recorded answer (structure, not quality)."""
    return {
        "answer_id": str(record.answer_id),
        "conversation_id": record.conversation_id,
        "created_at": record.created_at.isoformat(),
        "model_id": record.model_id,
        "development_double": record.model_is_development_double,
        "retrieval_strategy": record.provenance.get("retrieval_strategy", "unknown"),
        "planner_mode": record.provenance.get("planner_mode", "unknown"),
        "evidence_source_ids": [str(item) for item in record.evidence_source_ids],
        "cited_source_ids": [str(item) for item in record.cited_source_ids],
        "claim_count": record.claim_count,
        "correction_of": str(record.correction_of) if record.correction_of is not None else None,
    }


def _answers_list(store: SQLiteAnswerStore, args: argparse.Namespace) -> int:
    records = store.list(conversation_id=args.conversation_id)
    if args.json:
        payload = [_answer_brief(record) for record in records]
        print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))
        return 0
    if not records:
        scope = f" for conversation {args.conversation_id!r}" if args.conversation_id else ""
        print(f"no recorded answers{scope}")
        return 0
    for record in records:
        strategy = record.provenance.get("retrieval_strategy", "unknown")
        double = " (development double)" if record.model_is_development_double else ""
        print(
            f"{record.answer_id}  {record.created_at.isoformat()}  {strategy}{double}\n"
            f"   conversation={record.conversation_id} "
            f"evidence={len(record.evidence_source_ids)} "
            f"cited={len(record.cited_source_ids)} claims={record.claim_count}"
        )
    return 0


def _answers_show(store: SQLiteAnswerStore, args: argparse.Namespace) -> int:
    answer_id = _parse_answer_id(args.answer_id)
    if answer_id is None:
        return 1
    record = store.get(answer_id)
    if record is None:
        print(f"{PROGRAM}: no answer {answer_id} is recorded", file=sys.stderr)
        return 1
    if args.json:
        detail = {
            **_answer_brief(record),
            "text": record.text,
            "provenance": dict(record.provenance),
        }
        print(json.dumps(detail, sort_keys=True, indent=2, ensure_ascii=False))
        return 0
    print(_format_answer(record))
    return 0


def _answers_lineage(store: SQLiteAnswerStore, args: argparse.Namespace) -> int:
    answer_id = _parse_answer_id(args.answer_id)
    if answer_id is None:
        return 1
    # history() raises AnswerError for an unknown identifier, which the caller reports.
    chain = store.history(answer_id)
    if args.json:
        payload = [_answer_brief(record) for record in chain]
        print(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False))
        return 0
    print(f"correction lineage for {answer_id} ({len(chain)} answer(s), oldest first):")
    for record in chain:
        marker = f"corrects {record.correction_of}" if record.correction_of else "original"
        print(f"  {record.answer_id}  {record.created_at.isoformat()}  {marker}")
    return 0


def _parse_answer_id(raw: str) -> AnswerId | None:
    try:
        return AnswerId.parse(raw)
    except ChakasoError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return None


def _format_answer(record: AnswerRecord) -> str:
    double = (
        " (development double — not a language model)" if record.model_is_development_double else ""
    )
    evidence = ", ".join(str(item) for item in record.evidence_source_ids) or "(none)"
    cited = ", ".join(str(item) for item in record.cited_source_ids) or "(none)"
    lines = [
        f"answer: {record.answer_id}",
        f"conversation: {record.conversation_id}  turn_index: {record.turn_index}",
        f"created: {record.created_at.isoformat()}",
        f"model: {record.model_id}{double}",
    ]
    for key in sorted(record.provenance):
        lines.append(f"provenance.{key}: {record.provenance[key]}")
    lines.append(f"evidence sources: {evidence}")
    lines.append(f"cited sources: {cited}")
    lines.append(f"claims: {record.claim_count}")
    if record.correction_of is not None:
        lines.append(f"corrects: {record.correction_of}")
    lines.append(f"text: {record.text}")
    return "\n".join(lines)


def _format_config(loaded: LoadedConfig) -> str:
    key_width = max(len(item.key) for item in loaded.resolved)
    value_width = max(len(_format_value(item.value)) for item in loaded.resolved)

    lines = [f"{'KEY':<{key_width}}  {'VALUE':<{value_width}}  SOURCE"]
    for item in loaded.resolved:
        value = _format_value(item.value)
        lines.append(f"{item.key:<{key_width}}  {value:<{value_width}}  {item.source}")

    if not loaded.sources:
        lines.append("")
        lines.append("No configuration file was read; these are the built-in defaults.")
    return "\n".join(lines)


def _format_value(value: object) -> str:
    return repr(value) if isinstance(value, str) else str(value)


def _print_info() -> None:
    print(f"{PROGRAM} {__version__}")
    print(f"python {platform.python_version()} ({sys.implementation.name})")
    print(f"platform {platform.system()} {platform.release()} ({platform.machine()})")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
