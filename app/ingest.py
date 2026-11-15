"""Dual-source ingestion state machine (采集双源核验).

Collectors receive the same article over two independent crawl pipelines
(sources A and B). An article may enter the corpus only when the *latest*
reports from both pipelines carry identical original text and the same
train/test side, and neither pipeline has retracted it.

Per (article, source) the latest accepted report is stored; the article-level
verdict is derived from the two latest reports:

- any source's latest report is a retraction -> ``retracted`` (withdrawn)
- both content reports, same text and side   -> ``approved`` (in the corpus)
- both content reports, anything differs     -> ``conflict`` (withdrawn)
- only one source has reported content       -> ``pending`` (not yet in)

Revisions are monotonic per (article, source): a re-sent revision with
identical content is an idempotent no-op, the same revision with different
content and any older revision are rejected — so a late duplicate can never
resurrect an article that was since revised or retracted.
"""

# Article-level verdicts.
APPROVED = "approved"
PENDING = "pending"
CONFLICT = "conflict"
RETRACTED = "retracted"

# Report kinds stored per (article, source).
KIND_CONTENT = "content"
KIND_RETRACTION = "retraction"


def derive_ingest_state(rows):
    """Derive the verdict for one article from its stored source reports.

    *rows*: iterable of sqlite rows/dicts with keys ``source``, ``kind``,
    ``side``, ``text`` (the latest report per source, 0–2 rows).

    Returns ``(status, approved_side, approved_text, conflicts)`` where the
    approved fields are only set when the article may enter the corpus, and
    ``conflicts`` lists the differing fields ("text" / "side") for the
    conflict verdict.
    """
    by_source = {row["source"]: row for row in rows}
    a = by_source.get("A")
    b = by_source.get("B")

    if any(r["kind"] == KIND_RETRACTION for r in (a, b) if r is not None):
        return RETRACTED, None, None, []

    if a is not None and b is not None:
        conflicts = []
        if a["text"] != b["text"]:
            conflicts.append("text")
        if a["side"] != b["side"]:
            conflicts.append("side")
        if conflicts:
            return CONFLICT, None, None, conflicts
        return APPROVED, a["side"], a["text"], []

    return PENDING, None, None, []
