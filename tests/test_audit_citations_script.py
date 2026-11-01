"""Offline tests for the citation audit: extraction, feed parsing and the verdict rules.

The audit's value is that it fails on a citation that cannot be right, so these tests build the failures
deliberately — a wrong first author, a title that is another paper's, an identifier arXiv does not know — and
check that each is called what it is. Nothing here touches the network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "audit_citations.py"

_FEED = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2507.09404v2</id>
    <title>Scaling Laws for Optimal Data
      Mixtures</title>
    <published>2025-07-12T10:00:00Z</published>
    <author><name>Mustafa Shukor</name></author>
    <author><name>Louis Bethune</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2306.11644v2</id>
    <title>Textbooks Are All You Need</title>
    <published>2023-06-20T17:00:00Z</published>
    <author><name>Suriya Gunasekar</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors</id>
    <title>Error</title>
  </entry>
</feed>
"""


@pytest.fixture(scope="module")
def audit() -> ModuleType:
    spec = importlib.util.spec_from_file_location("audit_citations_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _verdicts(audit: ModuleType, text: str) -> list[str]:
    papers = audit.parse_feed(_FEED)
    return [row[1] for row in audit.audit(audit.extract_citations(text, "note.md"), papers)]


def test_a_feed_is_parsed_into_unversioned_papers_and_an_error_entry_is_not_a_paper(
    audit: ModuleType,
) -> None:
    papers = audit.parse_feed(_FEED)

    assert set(papers) == {"2507.09404", "2306.11644"}
    assert papers["2507.09404"].title == "Scaling Laws for Optimal Data Mixtures"
    assert papers["2507.09404"].authors == ("Mustafa Shukor", "Louis Bethune")
    assert papers["2507.09404"].published == "2025-07-12"


def test_every_citation_form_the_notes_use_is_extracted(audit: ModuleType) -> None:
    text = (
        "Shukor et al. (arXiv:2507.09404) fit mixture laws.\n"
        "See <https://arxiv.org/abs/2306.11644v2> and [arXiv:2306.11644](https://arxiv.org/abs/2306.11644).\n"
    )

    found = audit.extract_citations(text, "note.md")

    assert [(c.identifier, c.line) for c in found] == [
        ("2507.09404", 1),
        ("2306.11644", 2),
    ], "a Markdown link repeats its identifier in text and URL; that is one citation"
    assert found[0].named_author == "Shukor"


def test_a_named_first_author_who_is_the_first_author_passes(audit: ModuleType) -> None:
    assert _verdicts(audit, "Shukor et al. (arXiv:2507.09404) fit laws.") == ["author-ok"]


def test_a_named_author_who_did_not_write_the_paper_fails(audit: ModuleType) -> None:
    assert _verdicts(audit, "Hoffmann et al. (arXiv:2507.09404) fit laws.") == ["author-mismatch"]


def test_a_later_author_named_as_the_first_is_still_a_mismatch(audit: ModuleType) -> None:
    papers = audit.parse_feed(_FEED)
    citation = audit.extract_citations("Bethune et al. (arXiv:2507.09404) fit laws.", "n.md")[0]

    verdict, note = audit.judge(citation, papers)

    assert verdict == "author-mismatch"
    assert "later author" in note


def test_an_identifier_arxiv_does_not_know_is_unresolved(audit: ModuleType) -> None:
    assert _verdicts(audit, "see arXiv:1111.11111 for the proof.") == ["unresolved"]


def test_a_cited_italic_title_is_matched_loosely_and_another_papers_title_fails(
    audit: ModuleType,
) -> None:
    right = "Gunasekar, S. et al. *Textbooks Are All You Need.* arXiv:2306.11644, Jun 2023."
    wrong = "Gunasekar, S. et al. *Deep Residual Learning for Images.* arXiv:2306.11644, Jun 2023."

    assert _verdicts(audit, right) == ["author-ok"]
    # The wrong title is found because the *title* check runs when no author is named; here the reference
    # form names the author, so the author passes and the title is not consulted.
    assert _verdicts(audit, wrong) == ["author-ok"]
    anonymous_wrong = "*Deep Residual Learning for Images*, arXiv:2306.11644."
    assert _verdicts(audit, anonymous_wrong) == ["title-mismatch"]
    anonymous_right = "*Textbooks Are All You Need*, arXiv:2306.11644."
    assert _verdicts(audit, anonymous_right) == ["title-ok"]


def test_a_stated_date_corroborates_a_bare_identifier_and_a_different_one_asks_for_a_look(
    audit: ModuleType,
) -> None:
    assert _verdicts(audit, "the phi-1 paper arXiv:2306.11644, v1 20 June 2023.") == ["date-ok"]
    assert _verdicts(audit, "the phi-1 paper arXiv:2306.11644, v2 4 March 2024.") == [
        "date-differs"
    ]
    assert _verdicts(audit, "the phi-1 paper arXiv:2306.11644 describes it.") == ["unchecked"]


def test_only_a_failed_check_fails_the_audit(audit: ModuleType) -> None:
    assert {"unresolved", "author-mismatch", "title-mismatch"} == set(audit._FAILING)
    assert "date-differs" not in audit._FAILING


def test_the_audits_own_record_is_not_audited(audit: ModuleType) -> None:
    assert audit.is_audit_record("docs/internal/engineering/citation-audit-2026-11-01.md")
    assert not audit.is_audit_record("docs/research/open-model-landscape.md")


def test_a_cache_missing_an_identifier_asks_arxiv_for_only_that_one(
    audit: ModuleType, tmp_path: Path
) -> None:
    cache = tmp_path / "cache.json"
    asked: list[list[str]] = []

    def fetcher(identifiers: list[str]) -> dict[str, object]:
        asked.append(list(identifiers))
        return {key: value for key, value in audit.parse_feed(_FEED).items() if key in identifiers}

    first = audit.load_papers(["2507.09404"], cache, fetcher=fetcher)
    second = audit.load_papers(["2507.09404", "2306.11644"], cache, fetcher=fetcher)
    third = audit.load_papers(["2507.09404", "2306.11644"], cache, fetcher=fetcher)

    assert asked == [["2507.09404"], ["2306.11644"]], "the third call was served from the cache"
    assert set(first) == {"2507.09404"}
    assert set(second) == set(third) == {"2507.09404", "2306.11644"}
    assert audit.load_papers(["2507.09404"], cache, refresh=True, fetcher=fetcher)
    assert asked[-1] == ["2507.09404"], "refresh discards the cache"
