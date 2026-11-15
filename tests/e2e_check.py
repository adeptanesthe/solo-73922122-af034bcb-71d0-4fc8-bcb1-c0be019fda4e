#!/usr/bin/env python3
"""End-to-end requirement checks against two API processes sharing one DB.

Spawns two uvicorn processes on the same SQLite file (like the two compose
services), then verifies: input validation, batch atomicity, the 300-article
limit, 202 pending gating, contamination groups/edges, bridge-article group
splitting, modify-while-comparing (stale results discarded), cross-process
consistency, crash/restart recovery and no stale/mixed published results.

The dual-source ingestion API is covered as well: pending/approved/conflict/
retracted verdicts, monotonic revisions (idempotent replays, stale and
same-revision-different-content rejections), withdrawal on mismatch and
retraction, no resurrection by late reports, generation bumps only when the
approved corpus changes, manual/ingest namespace isolation, concurrent
cross-instance reporting and ingested articles flowing into /contamination.
"""

import concurrent.futures
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

import httpx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.demo import build_demo_articles  # noqa: E402

COMPARE_DELAY_MS = 1500
FAILURES = []


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


class Server:
    def __init__(self, port, db_path, workdir):
        self.port = port
        self.db_path = db_path
        # The app hard-exits when this file appears (chaos hook); used to
        # simulate crashes and to stop servers without signals.
        self.exit_file = os.path.join(workdir, f"exit-{port}")
        self.proc = None
        self.log = None

    @property
    def base(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        env = dict(
            os.environ,
            DB_PATH=self.db_path,
            INSTANCE_ID=f"local-{self.port}",
            COMPARE_DELAY_MS=str(COMPARE_DELAY_MS),
            WORKER_INTERVAL_MS="300",
            EXIT_WHEN_FILE_EXISTS=self.exit_file,
        )
        self.log = open(f"/tmp/corpus-e2e-{self.port}.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app",
             "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=REPO, env=env, stdout=self.log, stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 30
        while time.time() < deadline:
            try:
                if httpx.get(f"{self.base}/status", timeout=2).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.port} died; see {self.log.name}")
            time.sleep(0.2)
        raise RuntimeError(f"server {self.port} not ready in time")

    def kill(self):
        """Simulate a hard crash (the process os._exit()s on its own)."""
        if self.proc and self.proc.poll() is None:
            open(self.exit_file, "w").close()
            self.proc.wait(timeout=15)
        if os.path.exists(self.exit_file):
            os.unlink(self.exit_file)
        if self.log:
            self.log.close()


def wait_ready(base, timeout=60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            body = httpx.get(f"{base}/status", timeout=5).json()
            if body["up_to_date"]:
                return body
        except httpx.TransportError:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"{base} did not become up-to-date")


def main():
    workdir = tempfile.mkdtemp(prefix="corpus-e2e-")
    db_path = os.path.join(workdir, "corpus.db")
    s1 = Server(free_port(), db_path, workdir)
    s2 = Server(free_port(), db_path, workdir)
    s1.start()
    s2.start()
    base1, base2 = s1.base, s2.base
    print(f"two API processes up: {base1} {base2} (db={db_path})\n")

    try:
        run_checks(base1, base2, s1, s2)
        run_ingest_checks(base1, base2)
    finally:
        s1.kill()
        s2.kill()
        shutil.rmtree(workdir, ignore_errors=True)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES: {FAILURES}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


def run_checks(base1, base2, s1, s2):
    seed = build_demo_articles()

    # ---- input validation -------------------------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "abcd"}]})
    check("text shorter than 5 chars rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "a" * 2001}]})
    check("text longer than 2000 chars rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "valid", "text": "abcde"}]})
    check("side outside train/test rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "abcde"},
                                     {"id": "x", "side": "test", "text": "abcde"}]})
    check("duplicate ids inside one batch rejected (422)", r.status_code == 422)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "x", "side": "train", "text": "abcde"}],
                         "deletes": ["x"]})
    check("same id upserted and deleted rejected (422)", r.status_code == 422)
    r = httpx.get(f"{base1}/articles")
    check("no invalid article leaked into the corpus", r.json()["count"] == 0)

    # ---- batch atomicity ---------------------------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "good-1", "side": "train", "text": "valid text here"},
                                     {"id": "bad-1", "side": "train", "text": "no"}]})
    check("batch with one invalid article rejected", r.status_code == 422)
    r = httpx.get(f"{base1}/articles")
    check("failed batch applied nothing (atomicity)", r.json()["count"] == 0)

    # boundary lengths accepted
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "min5", "side": "train", "text": "abcde"},
                                     {"id": "max2000", "side": "test", "text": "z" * 2000}]})
    check("5-char and 2000-char texts accepted", r.status_code == 200)

    # ---- 300-article limit --------------------------------------------------
    big = [{"id": f"bulk-{i:03d}", "side": "train" if i % 2 else "test",
            "text": f"bulk article number {i} with enough text"} for i in range(298)]
    r = httpx.post(f"{base1}/articles/batch", json={"upserts": big})
    check("bulk upsert up to 300 articles accepted",
          r.status_code == 200 and r.json()["articles"] == 300)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "one-too-many", "side": "test",
                                      "text": "one article too many"}]})
    check("301st article rejected (409)", r.status_code == 409)
    r = httpx.get(f"{base1}/articles")
    check("corpus still exactly 300 after rejected batch", r.json()["count"] == 300)

    # clear corpus again (single atomic delete batch)
    ids = [a["id"] for a in r.json()["articles"]]
    r = httpx.post(f"{base1}/articles/batch", json={"deletes": ids})
    check("bulk delete accepted", r.status_code == 200 and r.json()["articles"] == 0)
    wait_ready(base1)

    # ---- contamination report across the two processes ---------------------
    r = httpx.post(f"{base1}/articles/batch", json={"upserts": seed})
    check("demo seed committed via api1", r.status_code == 200)
    gen_seed = r.json()["generation"]

    r = httpx.get(f"{base2}/contamination")
    check("api2 answers 202 while verification is pending",
          r.status_code == 202 and r.json()["status"] == "pending")

    wait_ready(base2)
    r = httpx.get(f"{base2}/contamination")
    report = r.json()
    check("api2 serves 200 once results cover the corpus",
          r.status_code == 200 and report["generation"] == gen_seed)
    check("two contamination groups found", report["contamination_group_count"] == 2)
    member_sets = {frozenset(g["train_members"] + g["test_members"]) for g in report["groups"]}
    check("direct pair grouped together",
          frozenset({"train-direct-1", "test-direct-1"}) in member_sets)
    check("bridge cluster grouped together",
          frozenset({"train-bridge-a", "test-bridge", "test-bridge-b"}) in member_sets)
    bridge_group = next(g for g in report["groups"] if "test-bridge" in g["test_members"])
    edge_pairs = {tuple(sorted((e["a"], e["b"]))) for e in bridge_group["edges"]}
    check("bridge group exposes direct similarity edges",
          edge_pairs == {("test-bridge", "train-bridge-a"), ("test-bridge", "test-bridge-b")},
          str(edge_pairs))
    check("every edge similarity >= 0.8",
          all(e["similarity"] >= 0.8 for g in report["groups"] for e in g["edges"]))

    # ---- deleting the bridge article splits the group ----------------------
    r = httpx.post(f"{base2}/articles/batch", json={"deletes": ["test-bridge"]})
    check("bridge deletion committed via api2", r.status_code == 200)
    wait_ready(base1)
    r = httpx.get(f"{base1}/contamination")
    report = r.json()
    grouped = {m for g in report["groups"] for m in g["train_members"] + g["test_members"]}
    check("bridge cluster split after deleting the bridge article",
          "train-bridge-a" not in grouped and "test-bridge-b" not in grouped)
    check("direct pair still reported", report["contamination_group_count"] == 1)

    # ---- modify while comparing: stale results must not overwrite ----------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "stale-probe", "side": "train",
                                      "text": "a completely fresh train article about nothing"}]})
    gen_a = r.json()["generation"]
    r = httpx.get(f"{base1}/contamination")
    check("202 right after a new batch (old groups not served with new corpus)",
          r.status_code == 202)
    # second batch lands while the worker computes generation gen_a
    r = httpx.post(f"{base2}/articles/batch",
                   json={"upserts": [{"id": "stale-probe-2", "side": "test",
                                      "text": "a completely fresh train article about nothink"}]})
    gen_b = r.json()["generation"]
    check("second batch accepted while background compare running",
          gen_b == gen_a + 1)
    final = wait_ready(base2)
    check("published generation caught up with the newest batch",
          final["published_generation"] == gen_b)
    r = httpx.get(f"{base2}/contamination")
    report = r.json()
    check("report serves exactly the newest generation (no stale/mixed results)",
          report["generation"] == gen_b)
    # the two probe articles differ by 1 char in 43 -> shingle jaccard ~0.74,
    # so they must NOT be linked; the point is the report reflects gen_b state.
    check("corpus content matches the newest batch",
          httpx.get(f"{base2}/articles").json()["count"] == 8)

    # ---- crash + restart recovery ------------------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "restart-a", "side": "train",
                                      "text": "restart probe article with some unique wording"},
                                     {"id": "restart-b", "side": "test",
                                      "text": "restart probe article with some unique wordinx"}]})
    gen_c = r.json()["generation"]
    s1.kill()  # crash both processes mid-verification
    s2.kill()
    time.sleep(0.5)
    s1.start()
    s2.start()
    final = wait_ready(base1, timeout=90)
    check("after crash+restart verification completes automatically",
          final["published_generation"] == gen_c)
    r = httpx.get(f"{base2}/contamination")
    report = r.json()
    check("post-restart report is consistent (single generation)",
          r.status_code == 200 and report["generation"] == gen_c)
    pairs = {tuple(sorted((e["a"], e["b"]))) for g in report["groups"] for e in g["edges"]}
    check("restart probe pair detected after recovery",
          ("restart-a", "restart-b") in pairs)

    # ---- clean verdict only when truly clean --------------------------------
    ids = [a["id"] for a in httpx.get(f"{base1}/articles").json()["articles"]]
    contaminated = [i for i in ids if i not in ("train-clean-1", "test-clean-1")]
    httpx.post(f"{base1}/articles/batch", json={"deletes": contaminated})
    r = httpx.get(f"{base1}/contamination")
    check("202 (not a premature clean verdict) right after deletions",
          r.status_code == 202)
    wait_ready(base1)
    r = httpx.get(f"{base1}/contamination")
    check("clean corpus reported as uncontaminated once verified",
          r.status_code == 200 and r.json()["contaminated"] is False
          and r.json()["groups"] == [])


def run_ingest_checks(base1, base2):
    """Dual-source ingestion (双源核验) requirement checks."""
    from scripts.demo import alter, pseudo_article
    import random

    def report(base, **payload):
        return httpx.post(f"{base}/ingest/reports", json=payload, timeout=10)

    def gen(base):
        return httpx.get(f"{base}/status", timeout=5).json()["current_generation"]

    def corpus_ids(base):
        return {a["id"] for a in httpx.get(f"{base}/articles", timeout=5).json()["articles"]}

    rng = random.Random(20261115)
    text_v1 = pseudo_article(rng)
    text_v2 = pseudo_article(rng)

    # ---- report validation -------------------------------------------------
    r = report(base1, source="A", id="ing-x", revision=1, retracted=True,
               text="retraction must not carry text")
    check("retraction carrying text rejected (422)", r.status_code == 422)
    r = report(base1, source="A", id="ing-x", revision=1, text="missing the side field")
    check("content report without side rejected (422)", r.status_code == 422)
    r = report(base1, source="A", id="ing-x", revision=0, side="train", text="revision zero")
    check("revision 0 rejected (422)", r.status_code == 422)
    r = report(base1, source="C", id="ing-x", revision=1, side="train", text="bad source")
    check("source outside A/B rejected (422)", r.status_code == 422)
    r = report(base1, source="A", id="ing-x", revision=1, side="train", text="tiny")
    check("ingest text shorter than 5 chars rejected (422)", r.status_code == 422)
    r = httpx.get(f"{base1}/ingest/articles/ing-x")
    check("unknown ingest id answers 404", r.status_code == 404)

    # ---- one source -> pending, no generation bump -------------------------
    g0 = gen(base1)
    r = report(base1, source="A", id="ing-1", revision=1, side="train", text=text_v1)
    body = r.json()
    check("first source report accepted as pending",
          r.status_code == 200 and body["status"] == "pending"
          and body["corpus_changed"] is False)
    check("pending report does not bump the corpus generation", gen(base1) == g0)
    check("pending article not in the corpus", "ing-1" not in corpus_ids(base1))

    # ---- matching second source (other instance) -> approved ---------------
    r = report(base2, source="B", id="ing-1", revision=1, side="train", text=text_v1)
    body = r.json()
    check("matching second source approves the article",
          r.status_code == 200 and body["status"] == "approved"
          and body["corpus_changed"] is True and body["generation"] == g0 + 1)
    arts = {a["id"]: a for a in httpx.get(f"{base2}/articles").json()["articles"]}
    check("approved article in corpus with origin=ingest",
          arts.get("ing-1", {}).get("origin") == "ingest"
          and arts["ing-1"]["side"] == "train" and arts["ing-1"]["text"] == text_v1)
    check("manual articles keep origin=manual",
          arts["train-clean-1"]["origin"] == "manual")

    # ---- idempotent replays: no generation bump -----------------------------
    g1 = gen(base1)
    r = report(base2, source="B", id="ing-1", revision=1, side="train", text=text_v1)
    check("same revision + same content replayed (no-op)",
          r.status_code == 200 and r.json()["replayed"] is True
          and r.json()["corpus_changed"] is False)
    r = report(base1, source="A", id="ing-1", revision=1, side="train", text=text_v1)
    check("cross-instance replay is a no-op too", r.status_code == 200
          and r.json()["replayed"] is True)
    check("replays do not advance the corpus generation", gen(base1) == g1)

    # ---- same revision different content / stale revision -------------------
    r = report(base1, source="A", id="ing-1", revision=1, side="test", text=text_v1)
    check("same revision, different side rejected (409)", r.status_code == 409)
    r = report(base1, source="B", id="ing-1", revision=1, side="train", text=text_v2)
    check("same revision, different text rejected (409)", r.status_code == 409)

    # ---- revision bump creates a conflict -> withdrawal ---------------------
    r = report(base1, source="A", id="ing-1", revision=2, side="train", text=text_v2)
    body = r.json()
    check("diverging new revision -> conflict, article withdrawn",
          r.status_code == 200 and body["status"] == "conflict"
          and body["corpus_changed"] is True)
    check("withdrawn article left the corpus", "ing-1" not in corpus_ids(base1))
    r = httpx.get(f"{base1}/ingest/articles/ing-1")
    body = r.json()
    check("conflict query shows both source versions",
          body["status"] == "conflict" and body["conflicts"] == ["text"]
          and body["sources"]["A"]["revision"] == 2
          and body["sources"]["B"]["revision"] == 1
          and body["sources"]["A"]["text"] == text_v2
          and body["sources"]["B"]["text"] == text_v1
          and body["in_corpus"] is False)

    # ---- late/stale reports cannot resurrect --------------------------------
    r = report(base1, source="A", id="ing-1", revision=1, side="train", text=text_v1)
    check("older revision rejected as stale (409)", r.status_code == 409)
    check("stale replay did not resurrect the article",
          "ing-1" not in corpus_ids(base1))

    # ---- second source converges -> approved again ---------------------------
    g2 = gen(base1)
    r = report(base2, source="B", id="ing-1", revision=2, side="train", text=text_v2)
    check("converging revision re-approves the article",
          r.status_code == 200 and r.json()["status"] == "approved"
          and r.json()["generation"] == g2 + 1)
    check("re-approved article back in the corpus", "ing-1" in corpus_ids(base1))

    # ---- retraction withdraws; late content cannot resurrect -----------------
    g3 = gen(base1)
    r = report(base1, source="A", id="ing-1", revision=3, retracted=True)
    check("retraction withdraws the approved article",
          r.status_code == 200 and r.json()["status"] == "retracted"
          and r.json()["corpus_changed"] is True
          and r.json()["generation"] == g3 + 1)
    check("retracted article out of the corpus", "ing-1" not in corpus_ids(base1))
    r = report(base1, source="A", id="ing-1", revision=2, side="train", text=text_v2)
    check("late pre-retraction content rejected (409)", r.status_code == 409)
    check("late report did not resurrect the retracted article",
          "ing-1" not in corpus_ids(base1))
    g4 = gen(base1)
    r = report(base1, source="A", id="ing-1", revision=3, retracted=True)
    check("retraction replay is an idempotent no-op",
          r.status_code == 200 and r.json()["replayed"] is True)
    r = report(base2, source="B", id="ing-1", revision=3, retracted=True)
    check("second source retracting changes nothing further",
          r.status_code == 200 and r.json()["corpus_changed"] is False)
    check("no recomputation while the approved corpus is unchanged",
          gen(base1) == g4)

    # ---- newer revisions after retraction can re-approve ---------------------
    r = report(base1, source="A", id="ing-1", revision=4, side="test", text=text_v1)
    check("content after retraction accepted, still blocked by other source",
          r.status_code == 200 and r.json()["status"] == "retracted"
          and r.json()["corpus_changed"] is False)
    r = report(base2, source="B", id="ing-1", revision=4, side="test", text=text_v1)
    check("both sources' new revisions approve again (side updated)",
          r.status_code == 200 and r.json()["status"] == "approved")
    arts = {a["id"]: a for a in httpx.get(f"{base1}/articles").json()["articles"]}
    check("re-approved article carries the new side",
          arts["ing-1"]["side"] == "test")

    # ---- manual/ingest namespace isolation -----------------------------------
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "ing-manual-1", "side": "train",
                                      "text": "a manually maintained article about health"}]})
    check("manual seed article committed", r.status_code == 200)
    r = report(base1, source="A", id="ing-manual-1", revision=1, side="train",
               text="a manually maintained article about health")
    check("ingestion must not overwrite a manual article (409)", r.status_code == 409)
    r = httpx.post(f"{base1}/articles/batch",
                   json={"upserts": [{"id": "ing-1", "side": "train",
                                      "text": "manual edit of an ingest-managed id"},
                                     {"id": "ing-manual-2", "side": "test",
                                      "text": "innocent bystander article about courts"}]})
    check("manual edit of ingest-managed id rejected (409)", r.status_code == 409)
    check("rejected batch applied nothing (atomicity)",
          "ing-manual-2" not in corpus_ids(base1))
    r = httpx.post(f"{base1}/articles/batch", json={"deletes": ["ing-1"]})
    check("manual delete of ingest-managed id rejected (409)", r.status_code == 409)
    check("ingest-managed article untouched by manual delete",
          "ing-1" in corpus_ids(base1))

    # ---- list endpoint --------------------------------------------------------
    r = httpx.get(f"{base2}/ingest/articles")
    summaries = {s["id"]: s for s in r.json()}
    check("ingest list shows tracked ids with verdicts",
          summaries.get("ing-1", {}).get("status") == "approved"
          and summaries["ing-1"]["revisions"] == {"A": 4, "B": 4}
          and summaries["ing-1"]["in_corpus"] is True)

    # ---- ingested near-duplicates flow into /contamination --------------------
    dup_base = pseudo_article(rng)
    dup_alt = alter(dup_base, [100], rng)
    for aid, side, text in [("ing-dup-a", "train", dup_base), ("ing-dup-b", "test", dup_alt)]:
        for source, base in [("A", base1), ("B", base2)]:
            r = report(base, source=source, id=aid, revision=1, side=side, text=text)
            assert r.status_code == 200, r.text
    r = httpx.get(f"{base1}/contamination")
    check("202 while ingest-driven changes await verification", r.status_code == 202)
    wait_ready(base2)
    r = httpx.get(f"{base2}/contamination")
    pairs = {tuple(sorted((e["a"], e["b"]))) for g in r.json()["groups"] for e in g["edges"]}
    check("ingested near-duplicate pair detected by verification",
          ("ing-dup-a", "ing-dup-b") in pairs, str(pairs))

    # ---- concurrent cross-instance reports settle consistently ----------------
    def post_report(base, payload):
        return httpx.post(f"{base}/ingest/reports", json=payload, timeout=30)

    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        futures = []
        for i in range(6):
            aid = f"ing-conc-{i}"
            text = pseudo_article(rng)
            futures.append(pool.submit(
                post_report, base1,
                {"source": "A", "id": aid, "revision": 1, "side": "train", "text": text}))
            futures.append(pool.submit(
                post_report, base2,
                {"source": "B", "id": aid, "revision": 1, "side": "train", "text": text}))
        responses = [f.result() for f in futures]
    check("all concurrent reports accepted",
          all(r.status_code == 200 for r in responses))
    per_article = {}
    for r in responses:
        per_article.setdefault(r.json()["id"], []).append(r.json()["status"])
    check("each concurrently reported article saw exactly one pending + one approved",
          all(sorted(v) == ["approved", "pending"] for v in per_article.values()),
          str(per_article))
    ids_now = corpus_ids(base2)
    check("all concurrently approved articles committed to the corpus",
          all(f"ing-conc-{i}" in ids_now for i in range(6)))
    wait_ready(base1)
    final = httpx.get(f"{base1}/status").json()
    check("verification caught up after concurrent ingest",
          final["up_to_date"] is True)
    check("status exposes ingest verdict counters",
          final["ingest"]["approved"] >= 8 and final["ingest"]["tracked_ids"] >= 8)


if __name__ == "__main__":
    main()
