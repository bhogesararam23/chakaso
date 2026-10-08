"""Citation validation: structural checks of claim-to-evidence references.

Validation asks whether each citation points at supplied evidence and, where a benchmark
declared the expected evidence, at the right evidence — and whether claims that had to be
cited were. It does not decide whether a source semantically *supports* a claim; that
needs an evaluator that does not exist, and this package is careful to say so.
"""

from __future__ import annotations

from chakaso.citation.metrics import CitationMetrics, citation_metrics
from chakaso.citation.validate import (
    CitationCheck,
    CitationStatus,
    ClaimCitationResult,
    validate_citations,
)

__all__ = [
    "CitationCheck",
    "CitationMetrics",
    "CitationStatus",
    "ClaimCitationResult",
    "citation_metrics",
    "validate_citations",
]
