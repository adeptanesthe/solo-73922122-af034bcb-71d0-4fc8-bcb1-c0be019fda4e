"""Corpus contamination checker.

Finds near-duplicate articles across a train/test corpus before model
evaluation. Two (or more) API processes share one SQLite database; every
committed batch modification bumps the corpus generation and enqueues a
background verification job. Verification results (direct similarity edges
and connected groups) are persisted per generation and published
atomically; queries never compare texts on the fly.

Articles enter the corpus two ways:

- manually, via atomic batches (POST /articles/batch), and
- by dual-source ingestion (POST /ingest/reports): collectors report each
  article from two independent crawl pipelines, and only when both
  pipelines' latest reports agree on text and train/test side (and neither
  retracted it) is the article approved into the corpus. Source versions,
  the approved projection and the verification job commit in one
  transaction; only changes to the approved corpus bump the generation.

The two namespaces are protected from each other: ingestion never
overwrites manual articles, and ingestion-managed ids reject manual edits.

Freshness rule: /contamination answers 200 only when the published results
cover the whole currently committed corpus (published_generation ==
current_generation); otherwise it answers 202 with the pending state, so
stale groups are never paired with new texts and a "clean" verdict is
never reported early.
"""

import logging
import os
import socket
import sqlite3
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from .db import MAX_ARTICLES, connect, get_meta, init_db
from .ingest import KIND_CONTENT, KIND_RETRACTION, derive_ingest_state
from .schemas import (
    BatchRequest,
    BatchResponse,
    IngestArticleStatus,
    IngestArticleSummary,
    IngestReport,
    IngestReportResponse,
    IngestSources,
    SourceVersion,
)
from .worker import CompareWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("corpus.api")

DB_PATH = os.environ.get("DB_PATH", "/data/corpus.db")
INSTANCE_ID = os.environ.get("INSTANCE_ID") or socket.gethostname()
WORKER_INTERVAL_MS = int(os.environ.get("WORKER_INTERVAL_MS", "1000"))
COMPARE_DELAY_MS = int(os.environ.get("COMPARE_DELAY_MS", "0"))
# Chaos/restart-demo knob: when set, the process hard-exits (os._exit, like
# a crash) as soon as this file appears. Inert when unset.
EXIT_WHEN_FILE_EXISTS = os.environ.get("EXIT_WHEN_FILE_EXISTS", "")


def _watch_exit_file(path: str) -> None:
    while True:
        if os.path.exists(path):
            os._exit(1)
        time.sleep(0.2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db(DB_PATH)
    worker = CompareWorker(
        DB_PATH,
        INSTANCE_ID,
        interval_ms=WORKER_INTERVAL_MS,
        compare_delay_ms=COMPARE_DELAY_MS,
    )
    # Restart recovery: if the corpus has unpublished changes (e.g. the
    # previous process died mid-verification), re-enqueue the job now.
    worker.ensure_pending_job()
    worker.start()
    app.state.worker = worker
    if EXIT_WHEN_FILE_EXISTS:
        log.warning("exit-file hook armed: %s", EXIT_WHEN_FILE_EXISTS)
        threading.Thread(
            target=_watch_exit_file, args=(EXIT_WHEN_FILE_EXISTS,), daemon=True
        ).start()
    log.info("instance %s ready, db=%s", INSTANCE_ID, DB_PATH)
    yield
    worker.stop()
    worker.join(timeout=5)


app = FastAPI(
    title="Corpus Contamination Checker",
    version="1.0.0",
    description=(
        "Detects near-duplicate articles (5-char shingle Jaccard >= 0.8) "
        "across train/test corpora. Results are computed in the background, "
        "persisted per corpus generation and served only when they cover "
        "the whole committed corpus."
    ),
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.post("/articles/batch", response_model=BatchResponse)
def apply_batch(req: BatchRequest):
    """Apply one atomic batch of additions/corrections/deletions.

    Every article is validated first; the batch is applied in a single
    transaction together with the generation bump and the verification job.
    Any failure (invalid article, 300-article limit, id conflicts) rolls
    the whole batch back — nothing takes effect partially.
    """
    upsert_ids = [u.id for u in req.upserts]
    if len(upsert_ids) != len(set(upsert_ids)):
        raise HTTPException(422, "duplicate article ids in upserts")
    if len(req.deletes) != len(set(req.deletes)):
        raise HTTPException(422, "duplicate article ids in deletes")
    conflict = set(upsert_ids) & set(req.deletes)
    if conflict:
        raise HTTPException(
            422, f"article ids both upserted and deleted: {sorted(conflict)}"
        )

    conn = connect(DB_PATH)
    try:
        if not req.upserts and not req.deletes:
            current, published = get_meta(conn)
            count = conn.execute("SELECT COUNT(*) AS c FROM articles").fetchone()["c"]
            return BatchResponse(
                changed=False,
                generation=current,
                published_generation=published,
                upserted=0,
                deleted=0,
                articles=count,
            )

        conn.execute("BEGIN IMMEDIATE")
        try:
            current, _ = get_meta(conn)
            new_generation = current + 1

            # Ids taken over by ingestion refuse manual edits. Checked inside
            # the write transaction so a concurrent ingest report on another
            # instance cannot slip past this guard.
            touched = sorted(set(upsert_ids) | set(req.deletes))
            placeholders = ",".join("?" for _ in touched)
            managed = conn.execute(
                f"SELECT DISTINCT article_id FROM ingest_reports"
                f" WHERE article_id IN ({placeholders})",
                touched,
            ).fetchall()
            if managed:
                raise HTTPException(
                    409,
                    "article ids managed by ingestion, manual edits refused: "
                    f"{sorted(r['article_id'] for r in managed)}",
                )

            deleted = 0
            for article_id in req.deletes:
                cur = conn.execute("DELETE FROM articles WHERE id = ?", (article_id,))
                deleted += cur.rowcount
            for u in req.upserts:
                conn.execute(
                    """
                    INSERT INTO articles (id, side, text, origin, updated_generation)
                    VALUES (?, ?, ?, 'manual', ?)
                    ON CONFLICT(id) DO UPDATE SET
                        side = excluded.side,
                        text = excluded.text,
                        updated_generation = excluded.updated_generation
                    """,
                    (u.id, u.side, u.text, new_generation),
                )

            count = conn.execute("SELECT COUNT(*) AS c FROM articles").fetchone()["c"]
            if count > MAX_ARTICLES:
                raise HTTPException(
                    409,
                    f"corpus limit exceeded: batch would leave {count} articles "
                    f"(max {MAX_ARTICLES}); whole batch rolled back",
                )

            conn.execute(
                "UPDATE meta SET current_generation = ? WHERE id = 1", (new_generation,)
            )
            conn.execute(
                "INSERT OR IGNORE INTO compute_jobs (generation, status)"
                " VALUES (?, 'pending')",
                (new_generation,),
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        _, published = get_meta(conn)
        log.info(
            "batch committed: generation=%d upserts=%d deleted=%d articles=%d",
            new_generation,
            len(req.upserts),
            deleted,
            count,
        )
        return BatchResponse(
            changed=True,
            generation=new_generation,
            published_generation=published,
            upserted=len(req.upserts),
            deleted=deleted,
            articles=count,
        )
    except HTTPException:
        raise
    except sqlite3.IntegrityError as exc:
        raise HTTPException(422, f"batch rejected, nothing applied: {exc}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# dual-source ingestion (双源核验)
# ---------------------------------------------------------------------------

def _ingest_rows(conn, article_id: str):
    return conn.execute(
        "SELECT article_id, source, revision, kind, side, text, updated_at"
        " FROM ingest_reports WHERE article_id = ?",
        (article_id,),
    ).fetchall()


def _source_version(row) -> SourceVersion:
    return SourceVersion(
        revision=row["revision"],
        kind=row["kind"],
        side=row["side"],
        text=row["text"],
        updated_at=row["updated_at"],
    )


def _ingest_status_model(article_id: str, rows, in_corpus: bool) -> IngestArticleStatus:
    status, approved_side, approved_text, conflicts = derive_ingest_state(rows)
    by_source = {r["source"]: r for r in rows}
    return IngestArticleStatus(
        id=article_id,
        status=status,
        in_corpus=in_corpus,
        conflicts=conflicts,
        sources=IngestSources(
            A=_source_version(by_source["A"]) if "A" in by_source else None,
            B=_source_version(by_source["B"]) if "B" in by_source else None,
        ),
        approved_side=approved_side,
        approved_text=approved_text,
    )


@app.post("/ingest/reports", response_model=IngestReportResponse)
def submit_ingest_report(req: IngestReport):
    """Report one article from one crawl pipeline (or retract it).

    The source version, the approved-corpus projection and the verification
    job commit in a single transaction, so concurrent reports on the two API
    instances always settle into one consistent state. Re-sending the same
    revision with identical content is an idempotent no-op (the corpus
    generation does not advance); the same revision with different content
    and older revisions are rejected, so a late report can never resurrect
    an article that was revised or retracted since.
    """
    kind = KIND_RETRACTION if req.retracted else KIND_CONTENT
    conn = connect(DB_PATH)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Manual articles are never touched by ingestion.
            article = conn.execute(
                "SELECT origin, side, text FROM articles WHERE id = ?", (req.id,)
            ).fetchone()
            if article is not None and article["origin"] == "manual":
                raise HTTPException(
                    409,
                    f"article {req.id!r} is managed manually; ingestion refused",
                )

            stored = conn.execute(
                "SELECT revision, kind, side, text FROM ingest_reports"
                " WHERE article_id = ? AND source = ?",
                (req.id, req.source),
            ).fetchone()
            if stored is not None:
                if req.revision < stored["revision"]:
                    raise HTTPException(
                        409,
                        f"stale revision {req.revision} for source {req.source} of "
                        f"{req.id!r}: latest accepted is {stored['revision']}",
                    )
                if req.revision == stored["revision"]:
                    identical = (
                        stored["kind"] == kind
                        and stored["side"] == req.side
                        and stored["text"] == req.text
                    )
                    if not identical:
                        raise HTTPException(
                            409,
                            f"revision {req.revision} for source {req.source} of "
                            f"{req.id!r} was already recorded with different content",
                        )
                    # Idempotent replay: nothing changes, no generation bump.
                    rows = _ingest_rows(conn, req.id)
                    status, _, _, _ = derive_ingest_state(rows)
                    current, published = get_meta(conn)
                    conn.commit()
                    return IngestReportResponse(
                        accepted=True,
                        replayed=True,
                        id=req.id,
                        source=req.source,
                        revision=req.revision,
                        status=status,
                        corpus_changed=False,
                        generation=current,
                        published_generation=published,
                    )

            conn.execute(
                """
                INSERT INTO ingest_reports
                    (article_id, source, revision, kind, side, text, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(article_id, source) DO UPDATE SET
                    revision = excluded.revision,
                    kind = excluded.kind,
                    side = excluded.side,
                    text = excluded.text,
                    updated_at = excluded.updated_at
                """,
                (req.id, req.source, req.revision, kind, req.side, req.text,
                 time.time()),
            )

            # Project the freshly derived verdict onto the corpus.
            rows = _ingest_rows(conn, req.id)
            status, approved_side, approved_text, _ = derive_ingest_state(rows)
            action = None  # None | "upsert" | "delete"
            if approved_side is not None:
                if article is None:
                    count = conn.execute(
                        "SELECT COUNT(*) AS c FROM articles"
                    ).fetchone()["c"]
                    if count + 1 > MAX_ARTICLES:
                        raise HTTPException(
                            409,
                            f"corpus limit exceeded: approving {req.id!r} would leave "
                            f"{count + 1} articles (max {MAX_ARTICLES}); "
                            "report rolled back",
                        )
                    action = "upsert"
                elif article["side"] != approved_side or article["text"] != approved_text:
                    action = "upsert"
            elif article is not None:
                # Verdict moved to pending/conflict/retracted: withdraw the
                # previously approved article (origin is 'ingest' here —
                # manual rows were rejected above).
                action = "delete"

            current, _ = get_meta(conn)
            generation = current
            corpus_changed = action is not None
            if corpus_changed:
                generation = current + 1
                if action == "upsert":
                    conn.execute(
                        """
                        INSERT INTO articles (id, side, text, origin, updated_generation)
                        VALUES (?, ?, ?, 'ingest', ?)
                        ON CONFLICT(id) DO UPDATE SET
                            side = excluded.side,
                            text = excluded.text,
                            origin = 'ingest',
                            updated_generation = excluded.updated_generation
                        """,
                        (req.id, approved_side, approved_text, generation),
                    )
                else:
                    conn.execute("DELETE FROM articles WHERE id = ?", (req.id,))
                conn.execute(
                    "UPDATE meta SET current_generation = ? WHERE id = 1",
                    (generation,),
                )
                conn.execute(
                    "INSERT OR IGNORE INTO compute_jobs (generation, status)"
                    " VALUES (?, 'pending')",
                    (generation,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

        _, published = get_meta(conn)
        log.info(
            "ingest report committed: id=%s source=%s revision=%d status=%s "
            "corpus_changed=%s generation=%d",
            req.id, req.source, req.revision, status, corpus_changed, generation,
        )
        return IngestReportResponse(
            accepted=True,
            replayed=False,
            id=req.id,
            source=req.source,
            revision=req.revision,
            status=status,
            corpus_changed=corpus_changed,
            generation=generation,
            published_generation=published,
        )
    except HTTPException:
        raise
    except sqlite3.IntegrityError as exc:
        raise HTTPException(422, f"report rejected, nothing applied: {exc}")
    finally:
        conn.close()


@app.get("/ingest/articles/{article_id}", response_model=IngestArticleStatus)
def get_ingest_article(article_id: str):
    """Both pipelines' latest versions of one article and the conflict state.

    This is what a collector inspects before re-crawling a mismatching or
    retracted article.
    """
    conn = connect(DB_PATH)
    try:
        rows = _ingest_rows(conn, article_id)
        if not rows:
            raise HTTPException(
                404, f"no ingestion state for article {article_id!r}"
            )
        in_corpus = (
            conn.execute(
                "SELECT 1 FROM articles WHERE id = ? AND origin = 'ingest'",
                (article_id,),
            ).fetchone()
            is not None
        )
    finally:
        conn.close()
    return _ingest_status_model(article_id, rows, in_corpus)


@app.get("/ingest/articles", response_model=list[IngestArticleSummary])
def list_ingest_articles():
    """All ingestion-managed article ids with their verdicts."""
    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT article_id, source, revision, kind, side, text, updated_at"
            " FROM ingest_reports ORDER BY article_id, source"
        ).fetchall()
        in_corpus = {
            r["id"]
            for r in conn.execute(
                "SELECT id FROM articles WHERE origin = 'ingest'"
            ).fetchall()
        }
    finally:
        conn.close()
    by_article: dict[str, list] = {}
    for row in rows:
        by_article.setdefault(row["article_id"], []).append(row)
    return [
        IngestArticleSummary(
            id=article_id,
            status=derive_ingest_state(article_rows)[0],
            in_corpus=article_id in in_corpus,
            revisions={r["source"]: r["revision"] for r in article_rows},
        )
        for article_id, article_rows in sorted(by_article.items())
    ]


@app.get("/articles")
def list_articles():
    """List the currently committed corpus."""
    conn = connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT id, side, text, origin, updated_generation FROM articles"
            " ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    return {
        "count": len(rows),
        "articles": [
            {
                "id": r["id"],
                "side": r["side"],
                "text": r["text"],
                "text_length": len(r["text"]),
                "origin": r["origin"],
                "updated_generation": r["updated_generation"],
            }
            for r in rows
        ],
    }


@app.get("/contamination")
def contamination():
    """Cross-set contamination report for the current corpus.

    200: results cover every committed article — returns contamination
    groups (connected components containing both train and test articles),
    their members and the direct similarity edges inside each group.
    202: verification is still pending for the current corpus state.
    """
    conn = connect(DB_PATH)
    try:
        # One read transaction => consistent view of pointer + results.
        conn.execute("BEGIN")
        current, published = get_meta(conn)
        if published != current:
            conn.rollback()
            return JSONResponse(
                status_code=202,
                content={
                    "status": "pending",
                    "detail": "verification does not cover the current corpus yet",
                    "current_generation": current,
                    "published_generation": published,
                },
                headers={"Retry-After": "1"},
            )
        group_rows = conn.execute(
            """
            SELECT g.group_id, g.article_id, g.side, a.text
            FROM article_groups g JOIN articles a ON a.id = g.article_id
            WHERE g.generation = ?
            ORDER BY g.group_id, g.article_id
            """,
            (published,),
        ).fetchall()
        edge_rows = conn.execute(
            "SELECT a, b, similarity FROM edges WHERE generation = ? ORDER BY a, b",
            (published,),
        ).fetchall()
        conn.commit()
    finally:
        conn.close()

    edges = [
        {"a": e["a"], "b": e["b"], "similarity": e["similarity"]} for e in edge_rows
    ]
    by_group = {}
    for r in group_rows:
        by_group.setdefault(r["group_id"], []).append(r)

    contamination_groups = []
    same_set_groups = 0
    for group_id, members in sorted(by_group.items()):
        sides = {m["side"] for m in members}
        if sides != {"train", "test"}:
            same_set_groups += 1
            continue
        member_ids = {m["article_id"] for m in members}
        contamination_groups.append(
            {
                "group_id": group_id,
                "size": len(members),
                "train_members": sorted(
                    m["article_id"] for m in members if m["side"] == "train"
                ),
                "test_members": sorted(
                    m["article_id"] for m in members if m["side"] == "test"
                ),
                "members": [
                    {"id": m["article_id"], "side": m["side"], "text": m["text"]}
                    for m in members
                ],
                "edges": [
                    e
                    for e in edges
                    if e["a"] in member_ids and e["b"] in member_ids
                ],
            }
        )

    return {
        "status": "ok",
        "generation": published,
        "contaminated": bool(contamination_groups),
        "contamination_group_count": len(contamination_groups),
        "same_set_group_count": same_set_groups,
        "groups": contamination_groups,
    }


@app.get("/status")
def status():
    """Instance, generation and background-job state."""
    conn = connect(DB_PATH)
    try:
        current, published = get_meta(conn)
        row = conn.execute(
            """
            SELECT COUNT(*) AS total,
                   COALESCE(SUM(side = 'train'), 0) AS train,
                   COALESCE(SUM(side = 'test'), 0) AS test,
                   COALESCE(SUM(origin = 'manual'), 0) AS manual,
                   COALESCE(SUM(origin = 'ingest'), 0) AS ingest
            FROM articles
            """
        ).fetchone()
        jobs = conn.execute(
            "SELECT generation, status, claimed_by, claimed_at FROM compute_jobs"
            " ORDER BY generation"
        ).fetchall()
        ingest_rows = conn.execute(
            "SELECT article_id, source, kind, side, text FROM ingest_reports"
        ).fetchall()
    finally:
        conn.close()

    by_article: dict[str, list] = {}
    for r in ingest_rows:
        by_article.setdefault(r["article_id"], []).append(r)
    verdicts: dict[str, int] = {}
    for article_rows in by_article.values():
        verdict = derive_ingest_state(article_rows)[0]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1

    return {
        "instance": INSTANCE_ID,
        "current_generation": current,
        "published_generation": published,
        "up_to_date": current == published,
        "articles": {
            "total": row["total"],
            "train": row["train"],
            "test": row["test"],
            "manual": row["manual"],
            "ingest": row["ingest"],
        },
        "ingest": {
            "tracked_ids": len(by_article),
            "approved": verdicts.get("approved", 0),
            "pending": verdicts.get("pending", 0),
            "conflict": verdicts.get("conflict", 0),
            "retracted": verdicts.get("retracted", 0),
        },
        "jobs": [
            {
                "generation": j["generation"],
                "status": j["status"],
                "claimed_by": j["claimed_by"],
                "claimed_at": j["claimed_at"],
            }
            for j in jobs
        ],
        "config": {
            "max_articles": MAX_ARTICLES,
            "compare_delay_ms": COMPARE_DELAY_MS,
            "worker_interval_ms": WORKER_INTERVAL_MS,
        },
    }
