"""Acquire a multi-register Project Gutenberg corpus under a declared selection rule.

The repository's first corpora were hand-picked short lists of novels, which is why K-022 records that the
only licensed corpus is one register and 6.7M tokens. This script replaces hand-picking with a rule that is
written down before any download happens and can be re-run to the same result:

* **Population.** Project Gutenberg's own catalogue rows whose ``Type`` is ``Text`` and whose ``Language``
  is ``en``. The catalogue snapshot's digest is recorded beside the selection.
* **Register.** The Library of Congress class the catalogue assigns: ``P`` prose, ``Q`` science, ``QA``
  mathematics (counted apart from the rest of ``Q``), ``T`` technology, ``R`` medicine. A work never
  appears in two registers.
* **Order.** Each register's eligible works are sorted by ebook number and shuffled with one declared seed.
  Nothing about the result depends on the order the catalogue file happened to list rows in.
* **Acceptance.** A download is accepted when its ingested text is between 20,000 and 2,000,000 characters.
  Below that a "book" is a pamphlet; above it one work would dominate a register. Every rejection is
  recorded with its reason, so the corpus is the rule's output rather than the operator's preference.
* **Politeness.** One request at a time, two seconds apart, with a declared user agent, which is the rate
  Project Gutenberg's robot policy asks for. Files already on disk are not fetched again.

The script writes raw bytes and a selection report into a work directory; the corpus catalogue that
:mod:`chakaso.training.dataset_catalog` pins is built from that report by ``--catalogue``. Neither the raw
bytes nor the report is committed: the catalogue carries the url and digest of every accepted source.

Usage:
    python scripts/acquire_gutenberg_corpus.py --catalog pg_catalog.csv --work data/corpora/NAME
    python scripts/acquire_gutenberg_corpus.py --work data/corpora/NAME --catalogue configs/corpora/NAME.toml
"""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.hashing import sha256_hex
from chakaso.training.dataset_catalog import (
    CatalogSource,
    DatasetCatalog,
    write_catalog,
)
from chakaso.training.source_corpus import (
    DEFAULT_MAX_CHARACTERS_PER_DOCUMENT,
    GUTENBERG_BASE_URL,
    ingest_project_gutenberg,
)

#: The order registers are listed in, which is also the order documents are packed in.
REGISTERS = ("prose", "science", "technology", "medicine", "mathematics")

#: Library of Congress prefixes per register. ``QA`` is checked before ``Q`` so mathematics is not science.
_LOC_CLASSES: dict[str, tuple[str, ...]] = {
    "prose": ("P",),
    "science": ("Q",),
    "technology": ("T",),
    "medicine": ("R",),
    "mathematics": ("QA",),
}

#: Declared acceptance bounds, in characters of ingested text.
MIN_CHARACTERS = 20_000
MAX_CHARACTERS = 2_000_000

#: Declared sampling seed. Changing it selects a different corpus, which would be a new catalogue version.
SELECTION_SEED = 20261030

#: Required spacing between requests, in seconds, per Project Gutenberg's robot policy.
REQUEST_SPACING_S = 2.0

USER_AGENT = "chakaso-corpus-acquisition/0.1 (offline language-model research; one request per 2s)"


def gutenberg_url_for(number: int) -> str:
    """The canonical Project Gutenberg text URL for one ebook number."""
    return f"{GUTENBERG_BASE_URL}{number}/pg{number}.txt"


@dataclass(frozen=True, slots=True)
class Candidate:
    number: int
    title: str
    author: str
    locc: str
    register: str


def _register_of(locc: str) -> str | None:
    cleaned = locc.strip().upper()
    if not cleaned:
        return None
    if cleaned.startswith("QA"):
        return "mathematics"
    for register in ("prose", "science", "technology", "medicine"):
        if cleaned.startswith(_LOC_CLASSES[register]):
            return register
    return None


def load_candidates(catalog_path: Path) -> dict[str, list[Candidate]]:
    """Every English text in the catalogue, bucketed by register, sorted by ebook number."""
    buckets: dict[str, list[Candidate]] = {name: [] for name in REGISTERS}
    with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("Type") != "Text" or row.get("Language") != "en":
                continue
            register = _register_of(row.get("LoCC") or "")
            if register is None:
                continue
            try:
                number = int(row["Text#"])
            except (KeyError, ValueError):
                continue
            author = (row.get("Authors") or "").strip()
            buckets[register].append(
                Candidate(
                    number=number,
                    title=(row.get("Title") or "").strip().replace("\n", " "),
                    author=" ".join(author.split()),
                    locc=(row.get("LoCC") or "").strip(),
                    register=register,
                )
            )
    for items in buckets.values():
        items.sort(key=lambda item: item.number)
    return buckets


def plan_selection(
    buckets: dict[str, list[Candidate]], quotas: dict[str, int], *, seed: int
) -> dict[str, list[Candidate]]:
    """The works the rule will try to download, per register, in walk order.

    The walk order is a seeded shuffle, but more works are planned than the quota needs, because a walk
    that stops at the first ``quota`` works cannot replace a rejected pamphlet. Acceptances fill the quota.
    """
    planned: dict[str, list[Candidate]] = {}
    rng = random.Random(seed)
    for register in REGISTERS:
        pool = list(buckets[register])
        rng.shuffle(pool)
        planned[register] = pool[: max(quotas.get(register, 0) * 4, quotas.get(register, 0))]
    return planned


def _fetch(number: int, destination: Path) -> str:
    url = gutenberg_url_for(number)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return sha256_hex(payload)


def acquire(
    planned: dict[str, list[Candidate]],
    quotas: dict[str, int],
    *,
    work: Path,
    spacing: float,
    verbose: bool,
) -> dict[str, object]:
    raw_dir = work / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    accepted: dict[str, int] = dict.fromkeys(REGISTERS, 0)
    records: list[dict[str, object]] = []
    # The same work can be catalogued twice (one edition per entry), so the rule also refuses a work it has
    # already accepted under a normalised title and author. Whole-work splits make a duplicate expensive.
    seen_works: set[tuple[str, str]] = set()
    for register in REGISTERS:
        quota = quotas.get(register, 0)
        if not quota:
            continue
        for candidate in planned[register]:
            if accepted[register] >= quota:
                break
            work_key = (" ".join(candidate.title.lower().split()), candidate.author.lower())
            if work_key in seen_works:
                records.append(
                    {
                        "slug": f"{register}-pg{candidate.number}",
                        "register": register,
                        "ebook_number": candidate.number,
                        "title": candidate.title,
                        "author": candidate.author,
                        "locc": candidate.locc,
                        "url": gutenberg_url_for(candidate.number),
                        "status": "rejected_duplicate_work",
                    }
                )
                continue
            slug = f"{register}-pg{candidate.number}"
            path = raw_dir / f"{slug}.txt"
            record: dict[str, object] = {
                "slug": slug,
                "register": register,
                "ebook_number": candidate.number,
                "title": candidate.title,
                "author": candidate.author,
                "locc": candidate.locc,
                "url": gutenberg_url_for(candidate.number),
            }
            if not path.is_file():
                try:
                    record["sha256"] = _fetch(candidate.number, path)
                    time.sleep(spacing)
                except (
                    urllib.error.URLError,
                    http.client.HTTPException,
                    TimeoutError,
                    OSError,
                ) as error:
                    record["status"] = "download_failed"
                    record["detail"] = f"{type(error).__name__}: {error}"
                    records.append(record)
                    time.sleep(spacing)
                    continue
            else:
                record["sha256"] = sha256_hex(path.read_bytes())
            raw_text = path.read_bytes().decode("utf-8-sig", errors="replace")
            try:
                ingested = ingest_project_gutenberg(
                    raw_text,
                    slug=slug,
                    ebook_number=candidate.number,
                    source_digest=str(record["sha256"]),
                )
            except Exception as error:
                record["status"] = "ingest_failed"
                record["detail"] = f"{type(error).__name__}: {error}"
                records.append(record)
                continue
            record["characters"] = len(ingested.text)
            record["header_title"] = ingested.title
            record["header_author"] = ingested.author
            if len(ingested.text) < MIN_CHARACTERS:
                record["status"] = "rejected_too_short"
                records.append(record)
                continue
            if len(ingested.text) > MAX_CHARACTERS:
                record["status"] = "rejected_too_long"
                records.append(record)
                continue
            record["status"] = "accepted"
            seen_works.add(work_key)
            accepted[register] += 1
            records.append(record)
            if verbose:
                print(
                    f"{register:<12} {accepted[register]:>3}/{quota}  pg{candidate.number:<6} "
                    f"{record['characters']:>9,}  {candidate.title[:52]}",
                    flush=True,
                )
    report = {
        "schema_version": "chakaso-corpus-selection/1",
        "selection_seed": SELECTION_SEED,
        "request_spacing_s": REQUEST_SPACING_S,
        "min_characters": MIN_CHARACTERS,
        "max_characters": MAX_CHARACTERS,
        "quotas": quotas,
        "accepted": accepted,
        "records": records,
    }
    (work / "selection.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def build_catalogue(
    report: dict[str, object], *, name: str, version: str, retrieved: str
) -> DatasetCatalog:
    """Turn the accepted records into the catalogue a training run pins.

    The slug's register prefix is what carries the register through the manifest: a document's path begins
    with its work's slug, so the register survives into every split and every report without a new schema
    field.
    """
    records = report["records"]
    assert isinstance(records, list)
    sources = []
    for record in records:
        if record.get("status") != "accepted":
            continue
        sources.append(
            CatalogSource(
                slug=str(record["slug"]),
                ebook_number=int(record["ebook_number"]),
                url=str(record["url"]),
                sha256=str(record["sha256"]),
                title=str(record.get("header_title") or record["title"]),
                author=str(record.get("header_author") or record["author"]),
            )
        )
    sources.sort(key=lambda item: item.slug)
    return DatasetCatalog(
        name=name,
        version=version,
        licence=(
            "Public domain in the United States; distributed by Project Gutenberg under the Project "
            "Gutenberg License, which permits copying, giving away and re-use; users outside the United "
            "States must check local law before use"
        ),
        source="Project Gutenberg (cache/epub text files, selection rule in scripts/acquire_gutenberg_corpus.py)",
        retrieved=retrieved,
        sources=tuple(sources),
        max_characters_per_document=DEFAULT_MAX_CHARACTERS_PER_DOCUMENT,
    )


def _parse_quotas(values: Sequence[str]) -> dict[str, int]:
    quotas = dict.fromkeys(REGISTERS, 0)
    for value in values:
        name, _, count = value.partition("=")
        if name not in quotas:
            message = f"unknown register {name!r}; expected one of {REGISTERS}"
            raise SystemExit(message)
        quotas[name] = int(count)
    return quotas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, help="the downloaded pg_catalog.csv")
    parser.add_argument(
        "--work", type=Path, required=True, help="work directory for raw files and report"
    )
    parser.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="REGISTER=N",
        help=f"accepted works per register, one flag each; registers {REGISTERS}",
    )
    parser.add_argument("--catalogue", type=Path, help="write the pinned catalogue here and exit")
    parser.add_argument("--name", default="gutenberg-multiregister-v1")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--retrieved", default=time.strftime("%Y-%m-%d"))
    parser.add_argument("--spacing", type=float, default=REQUEST_SPACING_S)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.catalogue is not None:
        report = json.loads((args.work / "selection.json").read_text(encoding="utf-8"))
        catalog = build_catalogue(
            report, name=args.name, version=args.version, retrieved=args.retrieved
        )
        write_catalog(catalog, args.catalogue, allow_overwrite=True)
        print(f"wrote {args.catalogue} with {len(catalog.sources)} sources")
        return 0

    if args.catalog is None:
        parser.error("--catalog is required unless --catalogue is given")
    quotas = _parse_quotas(args.quota or [f"{register}=0" for register in REGISTERS])
    buckets = load_candidates(args.catalog)
    planned = plan_selection(buckets, quotas, seed=SELECTION_SEED)
    report = acquire(planned, quotas, work=args.work, spacing=args.spacing, verbose=not args.quiet)
    accepted = report["accepted"]
    assert isinstance(accepted, dict)
    total = sum(int(value) for value in accepted.values())
    characters = sum(
        int(record["characters"])
        for record in report["records"]  # type: ignore[union-attr]
        if record.get("status") == "accepted"
    )
    print(f"accepted {total} works, {characters:,} characters")
    for register in REGISTERS:
        if quotas[register]:
            print(f"  {register:<12} {accepted[register]:>4}/{quotas[register]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
