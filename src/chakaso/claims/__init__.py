"""Claims: decomposing an answer into units that can be cited, checked and corrected.

Correction and grounding are defined over *claims*, not whole messages, so a claim is a
first-class record here: immutable, with a stable identifier within its answer, the evidence
it cites, and an evaluation status. The status is what the supplied evidence supports — it
is never a claim that something is true in the world.

Extraction, citation validation and grounding build on this and are added alongside it.
"""

from __future__ import annotations

from chakaso.claims.errors import ClaimError, InvalidClaimError
from chakaso.claims.model import Claim, ClaimId, ClaimStatus, derive_claim_id

__all__ = [
    "Claim",
    "ClaimError",
    "ClaimId",
    "ClaimStatus",
    "InvalidClaimError",
    "derive_claim_id",
]
