"""Follow-up reassessment: re-examine a prior answer when a later turn brings new evidence.

``docs/correction.md`` frames reassessment as triggered by a follow-up that arrives with better
evidence. This is the thin integration that makes that concrete *without* inventing new machinery:
it composes the evidence a reassessment should read — the pack the earlier answer rested on, plus
the new evidence now in hand — and hands the prior answer's claims to ``reassess``. It decides
nothing on its own; the decision stays with the boundaries it calls (ADR-0016).

The limits are the rest of the project's: there is no persistence, so the prior answer reaches here
as the claims a caller recorded, not a re-read chat log; and there is no language model, so no
wording is rewritten. Nothing in the runtime calls this automatically — a caller invokes it.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence

from chakaso.claims.model import Claim, ClaimId
from chakaso.core.identifiers import ChunkId, SourceId
from chakaso.correction.reassess import Reassessment, reassess
from chakaso.evidence import EvidencePack
from chakaso.evidence.records import EvidenceChunk, SourceRecord
from chakaso.grounding import GroundingEvaluator

__all__ = ["merge_evidence", "reassess_followup"]


def merge_evidence(*packs: EvidencePack) -> EvidencePack:
    """Union several evidence packs into one, de-duplicated and deterministic.

    Chunks are keyed by identifier, so the same evidence supplied twice is counted once, and each
    kept chunk retains the source that owns it. A source no chunk refers to is dropped, exactly as
    ``EvidencePack.of`` would drop it. Chunks are ordered by identifier, so the result does not
    depend on the order the packs were passed in — a follow-up must not change the evidence set
    merely by how it was assembled.
    """
    chunks_by_id: dict[ChunkId, EvidenceChunk] = {}
    sources_by_id: dict[SourceId, SourceRecord] = {}
    for pack in packs:
        for chunk in pack.chunks:
            chunks_by_id[chunk.chunk_id] = chunk
            source = pack.source_for(chunk.source_id)
            if source is not None:
                sources_by_id[source.source_id] = source
    ordered = tuple(chunks_by_id[key] for key in sorted(chunks_by_id, key=str))
    return EvidencePack.of(ordered, sources_by_id.values())


def reassess_followup(
    answer_id: str,
    prior_claims: Sequence[Claim],
    new_evidence: EvidencePack,
    *,
    standing_evidence: EvidencePack | None = None,
    grounding_evaluator: GroundingEvaluator | None = None,
    requires_citation: Collection[ClaimId] | None = None,
    expected_evidence: Mapping[ClaimId, Collection[ChunkId]] | None = None,
) -> Reassessment:
    """Reassess a prior answer's claims against the standing evidence plus the new evidence.

    ``standing_evidence`` is the pack the earlier answer was made from; omitting it reassesses
    against ``new_evidence`` alone, which is how a caller detects that support has been withdrawn
    rather than merely added to. The composition is the caller's semantic choice — this helper
    only merges the packs and delegates, so the answer-level decision is still ``reassess``'s.
    """
    composed = (
        merge_evidence(new_evidence)
        if standing_evidence is None
        else merge_evidence(standing_evidence, new_evidence)
    )
    return reassess(
        answer_id,
        prior_claims,
        composed,
        grounding_evaluator=grounding_evaluator,
        requires_citation=requires_citation,
        expected_evidence=expected_evidence,
    )
