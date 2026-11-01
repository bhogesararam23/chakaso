"""Audit the arXiv citations in the repository's tracked Markdown against arXiv's own metadata.

A research document that cites a paper by identifier is making two claims: that the identifier exists, and that
it is the paper the sentence says it is. The first is checkable by lookup and the second, partly, by comparing
the author the sentence names ("Shukor et al. (arXiv:2507.09404)") with the authors arXiv lists. Neither check
reads the paper, so neither verifies what the sentence says the paper *found*; that remains a reading task, and
the audit's job is to narrow it by removing the citations that fail before any reading.

The script prints one line per citation and exits non-zero if any identifier does not resolve or any named
first author is absent from the listed authors. ``--refresh`` queries arXiv (one request per 40 identifiers,
3 seconds apart, which is the interval its API terms ask for); without it the cached metadata is used.

Usage:
    python scripts/audit_citations.py --refresh
    python scripts/audit_citations.py --markdown docs/internal/engineering/citation-audit.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parent.parent
API = "https://export.arxiv.org/api/query"
BATCH = 40
SPACING_SECONDS = 3.0
USER_AGENT = "chakaso-citation-audit/0.1 (offline language-model research; contact via repository)"
_ATOM = {"a": "http://www.w3.org/2005/Atom"}

#: ``arXiv:2305.16264`` and ``arxiv.org/abs/2305.16264``, with an optional version suffix.
_IDENTIFIER = re.compile(
    r"(?:arXiv:|arxiv\.org/(?:abs|pdf)/)(\d{4}\.\d{4,5})(?:v\d+)?", re.IGNORECASE
)
#: A named first author directly before the identifier: ``Shukor et al. (arXiv:...``, ``Hoffmann et al., 2022,
#: arXiv:...`` or ``Kaplan et al. [arXiv:...``. A bare surname with no ``et al.`` is not trusted as a claim.
_NAMED_AUTHOR = re.compile(
    r"([A-Z][A-Za-z'2019\-]+(?:\s+(?:de|van|von|der|la|le)\s+[A-Z][A-Za-z'2019\-]+)?)\s+et\s+al\.?"
    r"[\s,(\[]*(?:\d{4}[\s,)\]]*)?(?:arXiv:|arxiv\.org)?$",
)


#: A reference-list author, ``Lin, S. et al.`` or ``Zhang, B. and Sennrich, R.``, directly before the title.
_LISTED_AUTHOR = re.compile(
    r"(?:^|[\s.\]])(?:\d+\.\s*)?([A-Z][A-Za-z'2019\-]+),\s+(?:[A-Z]\.\s*)+(?:et\s+al\.|and\b|&|,)[^*]{0,80}$"
)
#: An italicised title ending just before the identifier: ``*Mixtral of Experts*, [arXiv:...``.
_NAMED_TITLE = re.compile(
    r"\*{1,2}([^*\n]{8,200}?)\*{1,2}[\s.,:;\[\(<]*(?:https?://)?(?:arXiv:|arxiv\.org)?$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]{3,}")
#: Words that carry no identity in a title, so that "all you need" cannot make two papers look alike.
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "are",
        "all",
        "you",
        "from",
        "that",
        "this",
        "how",
        "why",
        "can",
        "via",
        "using",
        "our",
        "not",
    }
)
_MONTHS = {
    name: index
    for index, names in enumerate(
        [
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}
#: ``20 June 2023``, ``7 Oct 2024`` and ``Jan 2025``: a day is optional, a month and year are not.
_DATE = re.compile(r"\b(?:(\d{1,2})\s+)?([A-Z][a-z]{2,8})\.?,?\s+(\d{4})\b")


@dataclass(frozen=True, slots=True)
class Citation:
    identifier: str
    path: str
    line: int
    named_author: str | None
    named_title: str | None = None
    dates: tuple[tuple[int | None, int, int], ...] = ()


@dataclass(frozen=True, slots=True)
class Paper:
    identifier: str
    title: str
    authors: tuple[str, ...]
    published: str


_FAILING = frozenset({"unresolved", "author-mismatch", "title-mismatch"})


def tracked_markdown(root: Path = REPOSITORY) -> list[Path]:
    """Markdown files under version control, so a local scratch note is never audited as the project's."""
    listed = subprocess.run(
        ["git", "ls-files", "*.md"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return [root / name for name in listed if name]


def extract_citations(text: str, path: str) -> list[Citation]:
    """Every arXiv identifier in ``text`` with the first author the sentence names for it, when it names one."""
    found: list[Citation] = []
    for number, line in enumerate(text.splitlines(), start=1):
        on_this_line: set[str] = set()
        for match in _IDENTIFIER.finditer(line):
            # A Markdown link repeats its identifier in the text and the URL; that is one citation, not two.
            if match.group(1) in on_this_line:
                continue
            on_this_line.add(match.group(1))
            before = line[max(0, match.start() - 90) : match.start()]
            author = _NAMED_AUTHOR.search(before)
            title = _NAMED_TITLE.search(line[max(0, match.start() - 260) : match.start()])
            named = author.group(1) if author else None
            if named is None and title is not None:
                # A reference-list line names its author before the italic title, not before the identifier.
                listed = _LISTED_AUTHOR.search(line[: line.find("*" + title.group(1))])
                named = listed.group(1) if listed else None
            found.append(
                Citation(
                    identifier=match.group(1),
                    path=path,
                    line=number,
                    named_author=named,
                    named_title=title.group(1).strip(" .,:;") if title else None,
                    dates=_dates_in(line),
                )
            )
    return found


def _dates_in(line: str) -> tuple[tuple[int | None, int, int], ...]:
    """Every ``(day or None, month, year)`` the line states, in the forms the research notes use."""
    found = []
    for day, month, year in _DATE.findall(line):
        number = _MONTHS.get(month.lower())
        if number is not None:
            found.append((int(day) if day else None, number, int(year)))
    return tuple(found)


def parse_feed(xml_text: str) -> dict[str, Paper]:
    """Parse an arXiv Atom feed into papers keyed by unversioned identifier."""
    papers: dict[str, Paper] = {}
    for entry in ET.fromstring(xml_text).findall("a:entry", _ATOM):
        raw = (entry.findtext("a:id", default="", namespaces=_ATOM) or "").rsplit("/abs/", 1)[-1]
        identifier = re.sub(r"v\d+$", "", raw)
        title = " ".join((entry.findtext("a:title", default="", namespaces=_ATOM) or "").split())
        # arXiv answers an unknown identifier with an entry titled "Error" rather than an HTTP error.
        if not identifier or title.lower() == "error":
            continue
        authors = tuple(
            " ".join((name.text or "").split())
            for name in entry.findall("a:author/a:name", _ATOM)
            if name.text
        )
        published = (entry.findtext("a:published", default="", namespaces=_ATOM) or "")[:10]
        papers[identifier] = Paper(identifier, title, authors, published)
    return papers


def fetch(identifiers: Sequence[str]) -> dict[str, Paper]:
    papers: dict[str, Paper] = {}
    for start in range(0, len(identifiers), BATCH):
        chunk = identifiers[start : start + BATCH]
        query = urllib.parse.urlencode({"id_list": ",".join(chunk), "max_results": len(chunk)})
        request = urllib.request.Request(f"{API}?{query}", headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=60) as response:
            papers.update(parse_feed(response.read().decode("utf-8")))
        if start + BATCH < len(identifiers):
            time.sleep(SPACING_SECONDS)
    return papers


def _surname(full_name: str) -> str:
    return full_name.strip().split()[-1].lower() if full_name.strip() else ""


def judge(citation: Citation, papers: dict[str, Paper]) -> tuple[str, str]:
    """``(verdict, note)`` for one citation: ``unresolved``, ``author-mismatch``, ``author-ok`` or ``unchecked``.

    ``author-ok`` means the named surname is the first author's. A name that is only a later author is a
    mismatch for this purpose, because "X et al." conventionally names the first author.
    """
    paper = papers.get(citation.identifier)
    if paper is None:
        return "unresolved", "arXiv returned no entry for this identifier"
    if citation.named_author is None:
        if citation.named_title is not None:
            return _judge_title(citation.named_title, paper)
        return _judge_date(citation, paper)
    claimed = _surname(citation.named_author)
    if paper.authors and _surname(paper.authors[0]) == claimed:
        return "author-ok", paper.authors[0]
    others = [name for name in paper.authors[1:] if _surname(name) == claimed]
    if others:
        return (
            "author-mismatch",
            f"{citation.named_author} is a later author; first author is {paper.authors[0]}",
        )
    first = paper.authors[0] if paper.authors else "no authors listed"
    return (
        "author-mismatch",
        f"{citation.named_author} is not among the authors; first author is {first}",
    )


def title_overlap(cited: str, actual: str) -> float:
    """The share of the cited title's words (three letters or more) that the arXiv title also contains."""
    wanted = set(_WORD.findall(cited.lower())) - _STOPWORDS
    if not wanted:
        return 0.0
    return len(wanted & set(_WORD.findall(actual.lower())) - _STOPWORDS) / len(wanted)


def _judge_title(cited: str, paper: Paper) -> tuple[str, str]:
    """A cited title is allowed to be a shortened form of the real one, so the bar is overlap, not equality."""
    overlap = title_overlap(cited, paper.title)
    if overlap >= 0.6:
        return "title-ok", paper.title
    return (
        "title-mismatch",
        f"cited as {cited!r} but arXiv titles it {paper.title!r} (overlap {overlap:.2f})",
    )


def _judge_date(citation: Citation, paper: Paper) -> tuple[str, str]:
    """Fall back on the dates a line states: one equal to the first-version date supports the citation.

    A date that differs is reported as ``date-differs`` and does not fail the audit, because a note may quite
    properly cite a later version's date; it is a prompt to look, not a finding.
    """
    if not citation.dates:
        return "unchecked", "the line names no first author, no title and no date"
    year, month, day = (int(part) for part in paper.published.split("-"))
    for stated_day, stated_month, stated_year in citation.dates:
        if (stated_year, stated_month) == (year, month) and stated_day in (None, day):
            return "date-ok", f"first version {paper.published}"
    stated = "; ".join(f"{d or '?'}/{m}/{y}" for d, m, y in citation.dates)
    return "date-differs", f"arXiv's first version is {paper.published}; the line states {stated}"


def audit(
    citations: Iterable[Citation], papers: dict[str, Paper]
) -> list[tuple[Citation, str, str]]:
    return [(citation, *judge(citation, papers)) for citation in citations]


def _markdown(rows: list[tuple[Citation, str, str]], papers: dict[str, Paper]) -> str:
    counts: dict[str, int] = {}
    for _, verdict, _ in rows:
        counts[verdict] = counts.get(verdict, 0) + 1
    unique = sorted({citation.identifier for citation, _, _ in rows})
    lines = [
        "| Verdict | Count |",
        "| --- | ---: |",
        *(f"| {name} | {count} |" for name, count in sorted(counts.items())),
        "",
        f"{len(rows)} citations of {len(unique)} distinct identifiers.",
        "",
        "| Identifier | Published | First author | Title (from arXiv) |",
        "| --- | --- | --- | --- |",
    ]
    for identifier in unique:
        paper = papers.get(identifier)
        if paper is None:
            lines.append(f"| {identifier} | — | — | **not resolved by arXiv** |")
            continue
        first = paper.authors[0] if paper.authors else "—"
        lines.append(
            f"| [{identifier}](https://arxiv.org/abs/{identifier}) | {paper.published} | {first} | {paper.title} |"
        )
    for heading, verdicts in (
        ("Citations that failed a check", _FAILING),
        (
            "Citations to read before trusting (a stated date is not the first version's)",
            {"date-differs"},
        ),
    ):
        flagged = [row for row in rows if row[1] in verdicts]
        lines += ["", f"### {heading}", ""]
        if not flagged:
            lines.append("None.")
        for citation, verdict, note in flagged:
            lines.append(
                f"- `{citation.path}:{citation.line}` arXiv:{citation.identifier} — **{verdict}**: {note}"
            )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--cache", type=Path, default=REPOSITORY / ".scratch" / "arxiv-metadata.json"
    )
    parser.add_argument(
        "--refresh", action="store_true", help="query arXiv instead of reading the cache"
    )
    parser.add_argument(
        "--markdown", type=Path, help="also write the results as a Markdown table here"
    )
    args = parser.parse_args(argv)

    citations: list[Citation] = []
    for path in tracked_markdown():
        relative = path.relative_to(REPOSITORY).as_posix()
        citations.extend(extract_citations(path.read_text(encoding="utf-8"), relative))
    identifiers = sorted({citation.identifier for citation in citations})

    if args.refresh or not args.cache.is_file():
        papers = fetch(identifiers)
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(
            json.dumps(
                {key: asdict(paper) for key, paper in papers.items()}, indent=1, ensure_ascii=False
            ),
            encoding="utf-8",
        )
    else:
        raw = json.loads(args.cache.read_text(encoding="utf-8"))
        papers = {
            key: Paper(**{**value, "authors": tuple(value["authors"])})
            for key, value in raw.items()
        }

    rows = audit(citations, papers)
    for citation, verdict, note in rows:
        print(f"{verdict:<16} {citation.path}:{citation.line}  arXiv:{citation.identifier}  {note}")
    if args.markdown is not None:
        args.markdown.write_text(_markdown(rows, papers), encoding="utf-8")
    failed = sum(1 for _, verdict, _ in rows if verdict in _FAILING)
    print(
        f"{len(rows)} citations, {len(identifiers)} identifiers, {failed} failed", file=sys.stderr
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
