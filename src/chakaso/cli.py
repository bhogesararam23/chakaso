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
from pathlib import Path

from chakaso import __version__
from chakaso.answers import AnswerError, SQLiteAnswerStore
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
from chakaso.config import ConfigError, LoadedConfig, load_config
from chakaso.conversation import (
    Conversation,
    ConversationError,
    ConversationManager,
    Reply,
    new_conversation_id,
)
from chakaso.experiments import (
    BenchmarkKind,
    RetrievalStrategyReport,
    baseline_experiment,
    compare_retrieval_strategies,
    run_experiment,
)
from chakaso.models import GenerationParams, ModelError, UnknownModelError, create_model
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
        help="hold a conversation against the configured model",
        description=(
            "Hold a conversation. Replies come from the model selected in "
            "configuration, which today is a deterministic development double: fixed "
            "text that exercises the conversation plumbing, not answers. Nothing is "
            "retrieved and the conversation is not saved."
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


def _build_manager(loaded: LoadedConfig) -> ConversationManager:
    """Compose a model and a conversation into a manager.

    This is the application's composition root: the one place where configuration,
    the model registry and the conversation layer meet. It is wiring rather than
    behaviour, which is why it lives at the entry point instead of inside a
    component.
    """
    config = loaded.config
    return ConversationManager(
        create_model(config),
        Conversation(conversation_id=new_conversation_id()),
        params=GenerationParams(
            max_new_tokens=config.model.max_new_tokens,
            temperature=config.model.temperature,
        ),
        max_context_turns=config.model.max_context_turns,
    )


def _run_chat(args: argparse.Namespace) -> int:
    try:
        loaded = _load_or_report(args.config)
    except ConfigError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    try:
        manager = _build_manager(loaded)
    except UnknownModelError as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    # Everything a person needs in order not to mistake this for an assistant goes
    # to stderr, so that stdout carries the reply alone and stays scriptable.
    print(_engine_notice(manager), file=sys.stderr)

    if args.message is not None:
        return _answer_once(manager, args.message)

    return _converse(manager)


def _engine_notice(manager: ConversationManager) -> str:
    """State plainly what produced the replies, before any are produced."""
    metadata = manager.model_metadata
    lines = [f"engine: {metadata.model_id} ({metadata.display_name})"]
    if metadata.development_double:
        lines.append(
            "This is a development double, not a language model. It returns fixed text "
            "that exercises the conversation plumbing, and it does not answer anything."
        )
    lines.append(
        "No retrieval, no citations, no correction, and no confidence. "
        "The conversation is not saved and is lost when this process exits."
    )
    return "\n".join(lines)


def _answer_once(manager: ConversationManager, message: str) -> int:
    try:
        reply = manager.send(message)
    except (ConversationError, ModelError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(reply.text)
    _report_caveats(reply)
    return 0


def _converse(manager: ConversationManager) -> int:
    """Read messages until end of input or a quit command."""
    try:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            if text in _QUIT_COMMANDS:
                break

            try:
                reply = manager.send(text)
            except (ConversationError, ModelError) as exc:
                # A failed turn leaves the conversation unchanged, so the session can
                # continue rather than losing everything said so far.
                print(f"{PROGRAM}: {exc}", file=sys.stderr)
                continue

            print(f"{PROGRAM}> {reply.text}")
            _report_caveats(reply)
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        return 130
    return 0


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
    print(
        f"{PROGRAM}: choose a benchmark subject: 'retrieval', 'correction' or 'strategies'",
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
