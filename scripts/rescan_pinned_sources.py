"""Re-fetch a deterministic sample of a pinned corpus's URLs and report what moved, without touching the pin (K-031).

The catalogue pins each source's raw SHA-256 and `corpus prepare` refuses a mismatch, which is correct as far as it
goes: it proves the bytes on this machine are the bytes that were pinned, and says nothing about what the URL serves
now. The one drift found so far — `science-pg3489`, whose upstream `Last-Modified` moved the day after the pin — was
discovered by re-fetching all 430 URLs once, an act too expensive to repeat, which is how a check becomes an
assumption. This is the bounded version: a stratified sample chosen by hash before any request is made, strict limits
on time, redirects and response size, and a report of what each URL returns today.

**Nothing here replaces anything.** The fetched bytes go to a temporary directory that is deleted when the run ends;
the catalogue is opened read-only and its file digest is re-checked at the end and reported, so "the pin was not
touched" is a measurement in the artifact rather than a promise. Where a source's bytes moved, the catalogue's own
classifier says whether the *ingested text* moved with them — header-only drift is survivable, text drift is the kind
that would break every later comparison — and that classification reads the bytes fetched today, never the pinned
file.

One thing the sample cannot give: the response headers. `chakaso.retrieval.acquire.HttpFetcher` returns bytes, media
type and a retrieval timestamp, not `Last-Modified` or `ETag`, so drift here is judged by content digest alone. That
is the stronger test anyway — K-031 began precisely with a header that moved while the ingested text did not.

Usage:
    python scripts/rescan_pinned_sources.py --catalog configs/corpora/gutenberg-multiregister-v1.toml \\
        --per-register 4 --out docs/data/gutenberg-multiregister-v1-rescan-2026-11-04.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chakaso.core.hashing import sha256_hex
from chakaso.retrieval.acquire import (
    AcquisitionRequest,
    Fetcher,
    HttpFetcher,
    make_urllib_transport,
)
from chakaso.retrieval.errors import AcquisitionError
from chakaso.retrieval.policy import FetchPolicy
from chakaso.training.dataset_catalog import CatalogSource, DatasetCatalog, load_catalog

__all__ = ["RESCAN_POLICY", "RescanRow", "main", "register_of", "rescan", "select_sources"]

#: The limits a rescan runs under, declared here so the report can print them. A Gutenberg plain text is ~0.1-6 MB,
#: so 8 MiB refuses a redirect to something that is not the book without refusing any book; 20 seconds is longer
#: than any healthy request to this archive has taken.
RESCAN_POLICY = FetchPolicy(
    allowed_schemes=frozenset({"https"}),
    timeout_seconds=20.0,
    max_redirects=3,
    max_response_bytes=8 * 1024 * 1024,
    user_agent="chakaso-corpus-drift-rescan/1.0 (language-model research; verifying pinned corpus digests)",
)

#: Requests are spaced so a research rescan is not a load event on a public archive.
REQUEST_SPACING_SECONDS = 2.0

#: What a rescan cannot show, stated in the artifact rather than in a comment in the script.
LIMITATIONS = (
    "a sample of URLs cannot bound the drift of the whole catalogue, and an unchanged sample is not an all-clear "
    "over the pinned corpus. A URL that now 404s, redirects or serves different bytes does not invalidate the pin: "
    "the archive holds the pinned bytes and the catalogue's raw digest still selects them. Drift is judged by "
    "content digest, because the policy-enforcing fetcher returns bytes rather than response headers, so "
    "Last-Modified and ETag moves are invisible here by construction. The classification of a moved source is "
    "computed from the bytes fetched today, and this command opens the catalogue read-only and writes nothing "
    "outside a temporary directory."
)


@dataclass(frozen=True, slots=True)
class RescanRow:
    """What one pinned URL returned, and what that means against the digest the catalogue declared."""

    slug: str
    url: str
    ebook_number: int
    expected_sha256: str
    observed_sha256: str | None
    status: str
    bytes_fetched: int | None
    final_url: str | None
    media_type: str | None
    redirected: bool
    retrieved_at: str
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "url": self.url,
            "ebook_number": self.ebook_number,
            "expected_sha256": self.expected_sha256,
            "observed_sha256": self.observed_sha256,
            "status": self.status,
            "bytes_fetched": self.bytes_fetched,
            "final_url": self.final_url,
            "media_type": self.media_type,
            "redirected": self.redirected,
            "retrieved_at": self.retrieved_at,
            "detail": self.detail,
        }


def register_of(slug: str) -> str:
    """The register a source belongs to, read off its slug the way the corpus names its work groups.

    ``mathematics-pg16449`` is mathematics. A slug that does not follow the convention keeps its own leading
    segment rather than being pooled into a bucket the split does not use, because a sample that silently mixes
    registers is a sample whose strata are a lie.
    """
    head, separator, _ = slug.partition("-pg")
    return head if separator else slug


def select_sources(catalog: DatasetCatalog, *, per_register: int) -> tuple[CatalogSource, ...]:
    """A deterministic, register-stratified sample: each register's sources ranked by hash, the best ``per_register``.

    The rank is ``sha256("rescan|" + slug)``, so the sample is a function of the catalogue alone: the same file
    gives the same sample to any caller on any day, and nothing a rescan finds can influence what it looked at.
    A register with fewer sources than the quota contributes all it has, and the shortfall is visible in the row
    count rather than padded from another register.

    Raises:
        ValueError: ``per_register`` is not a positive integer.
    """
    if isinstance(per_register, bool) or not isinstance(per_register, int) or per_register < 1:
        message = f"per_register must be an integer >= 1, got {per_register!r}"
        raise ValueError(message)
    by_register: dict[str, list[CatalogSource]] = {}
    for item in catalog.sources:
        by_register.setdefault(register_of(item.slug), []).append(item)
    chosen: list[CatalogSource] = []
    for name in sorted(by_register):
        ranked = sorted(
            by_register[name],
            key=lambda item: hashlib.sha256(f"rescan|{item.slug}".encode()).hexdigest(),
        )
        chosen.extend(ranked[:per_register])
    return tuple(sorted(chosen, key=lambda item: item.slug))


def _classification(catalog: DatasetCatalog, item: CatalogSource, content: bytes) -> str:
    """What kind of difference the fetched bytes hold, from the catalogue's own classifier over a temporary copy.

    A one-source catalogue is built because `classify_drift` walks every declared source, and the other 429 would
    come back missing. The staging directory is created in the system temp area and removed here — never beside the
    pinned raw bytes, which is the whole safety property of this command.
    """
    staging = Path(tempfile.mkdtemp(prefix="chakaso-rescan-"))
    try:
        (staging / f"{item.slug}.txt").write_bytes(content)
        subset = replace(catalog, sources=(item,))
        found = subset.classify_drift(staging)
        return found.get(item.slug, "unclassified-not-reported-by-the-classifier")
    except OSError:
        return "unclassified-could-not-stage-the-bytes"
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def rescan(
    catalog: DatasetCatalog,
    sources: Sequence[CatalogSource],
    fetcher: Fetcher,
    policy: FetchPolicy,
    *,
    catalog_path: Path,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Fetch each sampled URL under ``policy`` and report every row against the digest the catalogue declared.

    ``fetcher`` is the retrieval layer's policy-enforcing fetcher, injected so the whole of this is testable
    without a network, and ``sleep`` is injected for the same reason. The report re-reads the catalogue file's
    digest at the end so a reader can see the pin was not modified.

    Raises:
        ValueError: the catalogue file disappears mid-run, which is a different corpus than the one sampled.
    """
    catalog_digest = sha256_hex(catalog_path.read_bytes())
    rows: list[RescanRow] = []
    for index, item in enumerate(sources):
        if index:
            sleep(REQUEST_SPACING_SECONDS)
        try:
            acquired = fetcher.fetch(AcquisitionRequest(url=item.url), policy)
        except AcquisitionError as error:
            rows.append(
                RescanRow(
                    slug=item.slug,
                    url=item.url,
                    ebook_number=item.ebook_number,
                    expected_sha256=item.sha256,
                    observed_sha256=None,
                    status="fetch-refused",
                    bytes_fetched=None,
                    final_url=None,
                    media_type=None,
                    redirected=False,
                    retrieved_at=now().isoformat(),
                    detail=str(error),
                )
            )
            continue
        observed = sha256_hex(acquired.content)
        status = (
            "unchanged"
            if observed == item.sha256
            else f"drifted-{_classification(catalog, item, acquired.content)}"
        )
        rows.append(
            RescanRow(
                slug=item.slug,
                url=item.url,
                ebook_number=item.ebook_number,
                expected_sha256=item.sha256,
                observed_sha256=observed,
                status=status,
                bytes_fetched=len(acquired.content),
                final_url=acquired.final_url,
                media_type=acquired.media_type,
                redirected=acquired.final_url != acquired.requested_url,
                retrieved_at=acquired.retrieved_at.isoformat(),
            )
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.status] = counts.get(row.status, 0) + 1
    after_digest = sha256_hex(catalog_path.read_bytes())
    return {
        "catalog": {
            "name": catalog.name,
            "version": catalog.version,
            "sources": len(catalog.sources),
            "licence": catalog.licence,
            "provider": catalog.source,
        },
        "catalog_file_sha256": catalog_digest,
        "catalog_retrieved": catalog.retrieved,
        "rescanned_at": datetime.now(UTC).isoformat(),
        "policy": {
            "allowed_schemes": sorted(policy.allowed_schemes),
            "timeout_seconds": policy.timeout_seconds,
            "max_redirects": policy.max_redirects,
            "max_response_bytes": policy.max_response_bytes,
            "user_agent": policy.user_agent,
            "request_spacing_seconds": REQUEST_SPACING_SECONDS,
        },
        "selection": {
            "rule": 'sha256("rescan|" + slug) per register, lowest ranks first',
            "registers": sorted({register_of(item.slug) for item in sources}),
            "sampled": len(sources),
            "per_register": max(
                (
                    sum(1 for item in sources if register_of(item.slug) == name)
                    for name in {register_of(item.slug) for item in sources}
                ),
                default=0,
            ),
        },
        "counts": counts,
        "rows": [row.as_dict() for row in rows],
        "catalog_untouched": catalog_digest == after_digest,
        "limitations": LIMITATIONS,
    }


def _fetcher() -> HttpFetcher:
    """The policy-enforcing fetcher over the standard-library transport — the only line that reaches the network."""
    return HttpFetcher(make_urllib_transport())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--per-register", type=int, default=4)
    parser.add_argument(
        "--slugs",
        default="",
        help="comma-separated slugs to rescan instead of the stratified sample, for following up one source",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(
            f"error: {args.out} already exists; a drift record is not replaced unasked, so a re-run "
            "writes a new dated file",
            file=sys.stderr,
        )
        return 2
    try:
        catalog = load_catalog(args.catalog)
        wanted = {part.strip() for part in args.slugs.split(",") if part.strip()}
        if wanted:
            unknown = sorted(wanted - {item.slug for item in catalog.sources})
            if unknown:
                message = f"the catalogue declares none of {unknown}"
                raise ValueError(message)
            sources = tuple(sorted(item for item in catalog.sources if item.slug in wanted))
        else:
            sources = select_sources(catalog, per_register=args.per_register)
        report = rescan(catalog, sources, _fetcher(), RESCAN_POLICY, catalog_path=args.catalog)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    summary = ", ".join(f"{count} {status}" for status, count in sorted(report["counts"].items()))
    print(
        f"{report['selection']['sampled']} of {len(catalog.sources)} sources, catalogue untouched "
        f"{report['catalog_untouched']}: {summary}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
