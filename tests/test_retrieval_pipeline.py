"""End-to-end integration of the retrieval pipeline with no mocks inside it.

These tests chain the real components — ingest, normalize, chunk, index, retrieve, assemble
a pack, resolve citations, run a turn — so that the seams between them are exercised, not
just each component alone. They are offline and deterministic, and they assert the
properties that make the pipeline trustworthy: identical runs produce identical
identifiers and ordering, irrelevant queries fabricate nothing, and a citation to
retrieved evidence resolves while one to evidence that was not supplied does not.

The model here is a test double, either the shipped deterministic double or a stub that
emits a known identifier. Nothing in this file asserts anything about answer quality:
there is no language model to have quality.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from chakaso.conversation import Conversation, ConversationManager, new_conversation_id
from chakaso.evidence import resolve_citations
from chakaso.models import (
    GenerationParams,
    GenerationResult,
    Message,
    ModelMetadata,
)
from chakaso.retrieval import Corpus, RetrievalService, ingest_file, ingest_text

SESSION = datetime(2026, 10, 8, 12, 0, tzinfo=UTC)

POLICY = """\
# Grievances

A grievance is a complaint about treatment.

## Timeline

Grievances must be filed within thirty days.
"""

NOTES = "The library closes at nine in the evening. It is open on Saturdays.\n"


class _CitingModel:
    """A model double that answers with a fixed string, used to exercise citation flow."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    @property
    def metadata(self) -> ModelMetadata:
        return ModelMetadata(
            model_id="citing-stub",
            display_name="Citing stub (not a language model)",
            context_window=512,
        )

    def generate(self, messages: Sequence[Message], params: GenerationParams) -> GenerationResult:
        return GenerationResult(text=self._answer, model_id="citing-stub")


def build_corpus(tmp_path: Path) -> Corpus:
    policy_file = tmp_path / "policy.md"
    policy_file.write_text(POLICY, encoding="utf-8")
    notes_file = tmp_path / "notes.txt"
    notes_file.write_text(NOTES, encoding="utf-8")

    corpus = Corpus()
    corpus.add_document(ingest_file(policy_file, now=lambda: SESSION))
    corpus.add_document(ingest_file(notes_file, now=lambda: SESSION))
    # A supplied-text source, mixed in with the file sources, exercised together.
    corpus.add_document(
        ingest_text("Budget appeals are reviewed quarterly.", label="budget", now=lambda: SESSION)
    )
    return corpus


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_the_whole_pipeline_retrieves_and_resolves(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    service = RetrievalService(corpus)

    outcome = service.search("grievance timeline", top_k=5)

    assert not outcome.pack.is_empty
    for chunk in outcome.pack.chunks:
        assert outcome.pack.source_for(chunk.source_id) is not None
    # A retrieved chunk resolves as a real citation; the pack is the gate.
    resolution = resolve_citations(
        f"Per the policy [{outcome.pack.chunks[0].chunk_id}].", outcome.pack
    )
    assert len(resolution.citations) == 1
    assert resolution.is_clean is True


def test_retrieved_evidence_drives_a_conversation_turn(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    outcome = RetrievalService(corpus).search("grievance", top_k=3)
    cited = outcome.pack.chunks[0]

    manager = ConversationManager(
        _CitingModel(f"The policy says so [{cited.chunk_id}]."),
        Conversation(conversation_id=new_conversation_id()),
        now=lambda: SESSION,
    )
    reply = manager.send("What does the grievance policy say?", evidence=outcome.pack)

    assert reply.citation_resolution.citations
    assert reply.has_unresolved_references is False
    assert reply.turn.cited_source_ids == (cited.source_id,)
    assert set(reply.turn.evidence_source_ids) == set(outcome.pack.sources)


def test_a_reference_to_evidence_not_retrieved_is_recorded_not_resolved(
    tmp_path: Path,
) -> None:
    corpus = build_corpus(tmp_path)
    outcome = RetrievalService(corpus).search("grievance", top_k=3)
    invented = "src_aaaaaaaaaaaaaaaa"

    manager = ConversationManager(
        _CitingModel(f"See [{invented}]."),
        Conversation(conversation_id="conv_e2e"),
        now=lambda: SESSION,
    )
    reply = manager.send("grievance", evidence=outcome.pack)

    assert reply.citation_resolution.citations == ()
    assert reply.citation_resolution.unknown_identifiers == (invented,)


# ---------------------------------------------------------------------------
# Determinism across repeated runs
# ---------------------------------------------------------------------------


def test_two_identical_runs_produce_identical_rankings(tmp_path: Path) -> None:
    first = RetrievalService(build_corpus(tmp_path)).search("library open", top_k=5)
    second = RetrievalService(build_corpus(tmp_path)).search("library open", top_k=5)

    assert [r.chunk.chunk_id for r in first.results] == [r.chunk.chunk_id for r in second.results]
    assert [round(r.score, 12) for r in first.results] == [
        round(r.score, 12) for r in second.results
    ]


def test_source_identifiers_are_stable_across_reingestion(tmp_path: Path) -> None:
    policy_file = tmp_path / "policy.md"
    policy_file.write_text(POLICY, encoding="utf-8")

    a = ingest_file(policy_file, now=lambda: SESSION)
    b = ingest_file(policy_file, now=lambda: SESSION)

    assert a.source.source_id == b.source.source_id
    assert [c.chunk_id for c in a.chunks] == [c.chunk_id for c in b.chunks]


# ---------------------------------------------------------------------------
# The awkward inputs the pipeline must define behaviour for
# ---------------------------------------------------------------------------


def test_an_irrelevant_query_yields_an_empty_pack(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    outcome = RetrievalService(corpus).search("quantum tunnelling", top_k=5)

    assert outcome.pack.is_empty is True
    assert outcome.results == ()


def test_a_duplicate_document_does_not_disturb_retrieval(tmp_path: Path) -> None:
    corpus = build_corpus(tmp_path)
    policy_file = tmp_path / "policy.md"
    before = RetrievalService(corpus).search("grievance", top_k=5)

    # Re-adding the same file is idempotent because identifiers are content-derived.
    corpus.add_document(ingest_file(policy_file, now=lambda: SESSION))
    after = RetrievalService(corpus).search("grievance", top_k=5)

    assert len(before.pack.chunks) == len(after.pack.chunks)


def test_an_empty_document_is_a_source_that_cannot_be_cited(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.write_text("", encoding="utf-8")
    document = ingest_file(empty, now=lambda: SESSION)

    corpus = Corpus()
    corpus.add_document(document)
    outcome = RetrievalService(corpus).search("anything", top_k=5)

    assert document.chunks == ()
    assert outcome.pack.is_empty is True


def test_unicode_content_is_retrievable(tmp_path: Path) -> None:
    unicode_file = tmp_path / "café.md"
    unicode_file.write_text("# Café\n\nLe résumé est naïve.\n", encoding="utf-8")

    corpus = Corpus()
    corpus.add_document(ingest_file(unicode_file, now=lambda: SESSION))
    outcome = RetrievalService(corpus).search("résumé naïve", top_k=5)

    assert not outcome.pack.is_empty
    assert "résumé" in outcome.pack.chunks[0].text.casefold()
