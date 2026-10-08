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
runs a retrieval or correction development benchmark over the shared synthetic fixtures and
prints a report that labels itself a development instrument; ``experiment run`` executes a
baseline experiment and prints its reproducible, content-fingerprinted result.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

from chakaso import __version__
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
from chakaso.experiments import BenchmarkKind, baseline_experiment, run_experiment
from chakaso.models import GenerationParams, ModelError, UnknownModelError, create_model
from chakaso.retrieval import (
    DEFAULT_TOP_K,
    Corpus,
    IngestionError,
    RetrievalError,
    RetrievalOutcome,
    RetrievalService,
    ingest_file,
)

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
    print(
        f"{PROGRAM}: choose a benchmark subject: 'retrieval' or 'correction'",
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
        runner = RetrievalBenchmarkRunner(RetrievalService(corpus), top_k=args.top_k)
        run = runner.run(dataset)
    except (BenchmarkError, IngestionError, RetrievalError) as exc:
        print(f"{PROGRAM}: {exc}", file=sys.stderr)
        return 1

    print(report_json(run) if args.json else format_report(run))
    return 0


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
