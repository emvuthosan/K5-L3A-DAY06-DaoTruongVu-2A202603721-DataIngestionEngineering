"""Day 06 pipeline entry point.

Run from starter/:  python -m src.run <command>

  ingest            read + normalize PDF pages, write manifest and ingest log
  verify-benchmark  validate benchmark/questions.json against normalized pages and freeze it (sha256)
  build             chunk (A fixed, B paragraph), embed, upsert into Chroma; chunk stats + timings
  reingest-check    rebuild without reset and prove there are no duplicates; test obsolete-chunk removal
  evaluate          run every question on A, B, C; save top-3 traces, Hit@1/Hit@3 by split, ANN check
  audit-sheets      write CSV sheets for manual coherence / metadata / grounding labels
  labels            merge results/audit/manual_labels.json into labeled CSVs + manual_summary.json
  all              ingest -> verify-benchmark -> build -> reingest-check -> evaluate -> audit-sheets
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "data" / "corpus_config.json"
MANIFEST = REPO / "data" / "manifest.json"
BENCHMARK = REPO / "benchmark" / "questions.json"
RESULTS = REPO / "results"
CHROMA_DIR = REPO / "chroma_db"

STRATEGIES: dict[str, dict[str, Any]] = {
    "A": {"chunking": "fixed_size", "chunk_size": 500, "overlap": 50, "collection": "A",
          "filter": "none", "hypothesis": "Baseline: fixed windows often cut sentences/evidence at boundaries."},
    "B": {"chunking": "by_paragraph", "chunk_size": 1000, "overlap": 0, "collection": "B",
          "filter": "none", "hypothesis": "Heading/paragraph-aware chunks keep ideas whole -> better coherence and Evidence Hit@3."},
    "C": {"chunking": "by_paragraph", "chunk_size": 1000, "overlap": 0, "collection": "B",
          "filter": "metadata `chapter` when the question provides one",
          "hypothesis": "Same chunks as B; scoping by chapter removes cross-chapter distractors on filter questions."},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- ingest
def load_pages() -> tuple[list, list[dict[str, Any]], dict[str, Any]]:
    from .ingestion import load_pdf

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    pdf = REPO / cfg["source_path"]
    if not pdf.exists():
        sys.exit(f"Missing {pdf}. Download it with the command in README.md (see data/manifest.json).")
    if _sha256(pdf) != cfg["sha256"]:
        sys.exit("PDF checksum differs from data/corpus_config.json - a different version was downloaded.")
    log: list[dict[str, Any]] = []
    pages = []
    for ch in cfg["chapters"]:
        pages += load_pdf(pdf, cfg["document_id"], cfg["source_path"], ch["chapter"], ch["title"],
                          ch["first_page"], ch["last_page"], log)
    return pages, log, cfg


def cmd_ingest(_: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    pages, log, cfg = load_pages()
    seconds = time.perf_counter() - t0
    _write_jsonl(RESULTS / "pages.jsonl", [
        {"document_id": p.document_id, "page": p.page, **p.metadata, "text": p.text,
         "paragraphs": [asdict(x) for x in p.paragraphs]} for p in pages])
    content = [p for p in pages if p.metadata["is_content_page"]]
    manifest = {
        "source_id": cfg["document_id"],
        "title": cfg["title"],
        "source_path_or_url": {"local": cfg["source_path"], "url": cfg["source_url"], "landing_page": cfg["landing_page"]},
        "license_or_permission": cfg["license_or_permission"],
        "accessed_at": cfg["accessed_at"],
        "snapshot": {"sha256": cfg["sha256"], "http_last_modified": cfg["http_last_modified"],
                     "physical_pages_in_file": cfg["physical_pages_in_file"]},
        "page_numbering": cfg["page_numbering"],
        "content_page_rule": cfg["content_page_rule"],
        "chapters": cfg["chapters"],
        "pages_ingested": len(pages),
        "content_pages": len(content),
        "total_normalized_chars": sum(p.metadata["n_chars"] for p in pages),
        "pages": [{"page_id": f"{p.document_id}:p{p.page:04d}", "page": p.page,
                   "printed_page": p.metadata["printed_page"], "chapter": p.metadata["chapter"],
                   "n_chars": p.metadata["n_chars"], "n_images": p.metadata["n_images"],
                   "counted_as_content": p.metadata["is_content_page"]} for p in pages],
        "generated_at": _now(),
    }
    _write_json(MANIFEST, manifest)
    _write_json(RESULTS / "ingest_log.json", {"seconds": round(seconds, 2), "events": log})
    print(f"Ingested {len(pages)} pages ({len(content)} content pages, "
          f"{manifest['total_normalized_chars']} chars) in {seconds:.1f}s; {len(log)} log events")
    if len(content) < 30:
        sys.exit("ERROR: fewer than 30 content pages")


def _pages_from_results():
    from .models import DocumentPage, Paragraph

    rows = _read_jsonl(RESULTS / "pages.jsonl")
    pages = []
    for r in rows:
        meta = {k: r[k] for k in ("chapter", "chapter_title", "printed_page", "n_images", "n_chars", "is_content_page")}
        pages.append(DocumentPage(r["document_id"], MANIFEST_SOURCE(), r["page"], r["text"], meta,
                                  [Paragraph(**p) for p in r["paragraphs"]]))
    return pages


def MANIFEST_SOURCE() -> str:
    return json.loads(CONFIG.read_text(encoding="utf-8"))["source_path"]


# ---------------------------------------------------------------- benchmark
def load_benchmark() -> list[dict[str, Any]]:
    return json.loads(BENCHMARK.read_text(encoding="utf-8"))["questions"]


def cmd_verify_benchmark(_: argparse.Namespace) -> None:
    from .evaluation import _norm, gold_options

    pages = {(r["document_id"], r["page"]): r for r in _read_jsonl(RESULTS / "pages.jsonl")}
    questions = load_benchmark()
    problems, rows = [], []
    ids = [q["id"] for q in questions]
    if len(set(ids)) != len(ids):
        problems.append("duplicate question ids")
    for q in questions:
        for opt in gold_options(q):
            page = pages.get((opt["source_id"], int(opt["page"])))
            ok = bool(page) and _norm(opt["evidence_quote"]) in _norm(page["text"])
            rows.append({"id": q["id"], "page": opt["page"], "evidence_found": ok})
            if not ok:
                problems.append(f"{q['id']}: evidence not found on page {opt['page']}")
        if q.get("filter"):
            for key, value in q["filter"].items():
                if key != "chapter" or not any(str(p["chapter"]) == str(value) for p in pages.values()):
                    problems.append(f"{q['id']}: filter {key}={value} not present in metadata")
    split = {s: sum(q["split"] == s for q in questions) for s in ("dev", "test")}
    filt = {s: sum(bool(q.get("filter")) and q["split"] == s for q in questions) for s in ("dev", "test")}
    if len(questions) < 20 or split["dev"] < 15 or split["test"] < 5:
        problems.append(f"need >=20 questions with >=15 dev / >=5 test, got {split}")
    if sum(filt.values()) < 3 or filt["test"] < 1:
        problems.append(f"need >=3 filter questions with >=1 in test, got {filt}")
    report = {"checked_at": _now(), "questions_sha256": _sha256(BENCHMARK), "n_questions": len(questions),
              "split_counts": split, "filter_counts": filt, "evidence_checks": rows, "problems": problems}
    _write_json(REPO / "benchmark" / "verification.json", report)
    print(f"{len(questions)} questions {split}, filter questions {filt}; sha256={report['questions_sha256'][:12]}")
    if problems:
        print("\n".join(problems))
        sys.exit("Benchmark verification FAILED")
    print("Benchmark verification OK")


# ---------------------------------------------------------------- build
def _embedder():
    from .embeddings import SentenceTransformerEmbedder
    return SentenceTransformerEmbedder()


def _store(embedder):
    from .store import ChromaStore
    return ChromaStore(CHROMA_DIR, embedder.dimensions, f"{embedder.model_name}@{embedder.revision}")


def make_chunks(strategy: str, pages) -> list:
    from .chunking import by_paragraph, fixed_size

    cfg = STRATEGIES[strategy]
    out = []
    for page in pages:
        if cfg["chunking"] == "fixed_size":
            out += fixed_size(page, cfg["chunk_size"], cfg["overlap"], strategy)
        else:
            out += by_paragraph(page, cfg["chunk_size"], strategy)
    return out


def _length_stats(values: list[int]) -> dict[str, float]:
    qs = statistics.quantiles(values, n=10)
    return {"min": min(values), "p10": round(qs[0], 1), "median": statistics.median(values),
            "mean": round(statistics.mean(values), 1), "p90": round(qs[-1], 1), "max": max(values)}


def cmd_build(args: argparse.Namespace) -> None:
    pages = _pages_from_results()
    embedder = _embedder()
    store = _store(embedder)
    stats: dict[str, Any] = {"built_at": _now(), "embedding": embedder.describe(),
                             "vector_store": {"engine": "chromadb", "version": __import__("chromadb").__version__,
                                              "path": str(CHROMA_DIR.relative_to(REPO)), "metric": "cosine",
                                              "index": "HNSW (approximate); see results/ann_check.json"},
                             "strategies": {}}
    for strategy in ("A", "B"):
        if getattr(args, "reset", False):
            store.reset(strategy)
        t0 = time.perf_counter()
        chunks = make_chunks(strategy, pages)
        t1 = time.perf_counter()
        vectors = embedder.embed_documents([c.text for c in chunks])
        t2 = time.perf_counter()
        upsert = store.upsert(strategy, chunks, vectors)
        t3 = time.perf_counter()
        tokens = [embedder.token_count(c.text) for c in chunks]
        lengths = [len(c.text) for c in chunks]
        stats["strategies"][strategy] = {
            **STRATEGIES[strategy], "chunk_count": len(chunks), "chunk_chars": _length_stats(lengths),
            "chunk_tokens": _length_stats(tokens), "total_chars": sum(lengths), "total_tokens": sum(tokens),
            "chunks_over_max_seq_length": sum(t > embedder.max_seq_length for t in tokens),
            "seconds": {"chunk": round(t1 - t0, 3), "embed": round(t2 - t1, 2), "upsert": round(t3 - t2, 2)},
            "cost_usd": 0.0, "upsert": upsert}
        _write_jsonl(RESULTS / f"chunks_{strategy}.jsonl",
                     [{"chunk_id": c.chunk_id, "page": c.page, **c.metadata, "n_chars": len(c.text),
                       "n_tokens": t, "text": c.text} for c, t in zip(chunks, tokens)])
        print(f"{strategy}: {len(chunks)} chunks, embed {t2 - t1:.1f}s, store now {upsert['after']}")
    stats["strategies"]["C"] = {**STRATEGIES["C"], "note": "reuses collection B (identical chunks and vectors)"}
    stats["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                            "torch": __import__("torch").__version__,
                            "sentence_transformers": __import__("sentence_transformers").__version__,
                            "pymupdf": __import__("fitz").VersionBind}
    _write_json(RESULTS / "build_stats.json", stats)


def cmd_reingest_check(_: argparse.Namespace) -> None:
    from .store import ChromaStore

    pages = _pages_from_results()
    embedder = _embedder()
    store = _store(embedder)
    report: dict[str, Any] = {"checked_at": _now(), "same_version": {}}
    for strategy in ("A", "B"):
        chunks = make_chunks(strategy, pages)
        before = store.count(strategy)
        res = store.upsert(strategy, chunks, embedder.embed_documents([c.text for c in chunks]))
        ids = store.collection(strategy).get(include=[])["ids"]
        report["same_version"][strategy] = {"count_before": before, "count_after": res["after"],
                                            "expected": len(chunks), "unique_ids": len(set(ids)),
                                            "deleted_obsolete": res["deleted_obsolete"],
                                            "ok": before == res["after"] == len(chunks) == len(set(ids))}
    # changed-document simulation on a scratch collection: drop the last page, re-ingest
    scratch = "reingest_test"
    store.reset(scratch)
    chunks = make_chunks("B", pages)
    vecs = embedder.embed_documents([c.text for c in chunks])
    store.upsert(scratch, chunks, vecs)
    last_page = max(c.page for c in chunks)
    kept = [(c, v) for c, v in zip(chunks, vecs) if c.page != last_page]
    res = store.upsert(scratch, [c for c, _ in kept], [v for _, v in kept])
    removed = len(chunks) - len(kept)
    report["changed_document_simulation"] = {
        "scenario": f"document re-ingested without page {last_page}",
        "chunks_initial": len(chunks), "chunks_after": res["after"], "deleted_obsolete": res["deleted_obsolete"],
        "expected_deleted": removed, "ok": res["deleted_obsolete"] == removed and res["after"] == len(kept)}
    store.reset(scratch)
    _write_json(RESULTS / "reingest_check.json", report)
    print(json.dumps(report, indent=2))
    if not all(v["ok"] for v in report["same_version"].values()) or not report["changed_document_simulation"]["ok"]:
        sys.exit("Re-ingest check FAILED")


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(_: argparse.Namespace) -> None:
    import numpy as np

    from .evaluation import evaluate_query, summarize

    questions = load_benchmark()
    verification = json.loads((REPO / "benchmark" / "verification.json").read_text(encoding="utf-8"))
    if verification["questions_sha256"] != _sha256(BENCHMARK) or verification["problems"]:
        sys.exit("Benchmark changed or failed verification since freezing - run verify-benchmark first")
    embedder = _embedder()
    store = _store(embedder)
    qvecs = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    traces, ann_rows = [], []
    records = {}
    for strategy in ("A", "B"):
        rec = store.all_records(strategy)
        records[strategy] = (rec["ids"], np.asarray(rec["embeddings"]), rec["metadatas"])
    t0 = time.perf_counter()
    for strategy, cfg in STRATEGIES.items():
        for q in questions:
            applied = q.get("filter") if strategy == "C" else None
            hits = store.search(cfg["collection"], qvecs[q["id"]], top_k=3, filters=applied)
            traces.append(evaluate_query(q, hits, strategy, applied))
            ids, mat, metas = records[cfg["collection"]]
            mask = np.array([all(str(m.get(k)) == str(v) for k, v in (applied or {}).items()) for m in metas])
            sims = mat @ np.asarray(qvecs[q["id"]])
            sims[~mask] = -np.inf
            exact = [ids[i] for i in np.argsort(-sims)[:3]]
            ann_rows.append({"strategy": strategy, "question_id": q["id"],
                             "same_top3_set": set(exact) == {h["chunk_id"] for h in hits},
                             "same_top3_order": exact == [h["chunk_id"] for h in hits]})
    seconds = time.perf_counter() - t0
    for strategy in STRATEGIES:
        _write_jsonl(RESULTS / "traces" / f"{strategy}.jsonl", [t for t in traces if t["strategy"] == strategy])
    summary = summarize(traces)
    _write_json(RESULTS / "summary.json", {"evaluated_at": _now(), "query_seconds_total": round(seconds, 2),
                                           "questions_sha256": verification["questions_sha256"], "rows": summary})
    _write_json(RESULTS / "ann_check.json", {
        "description": "Chroma HNSW top-3 vs exact cosine (numpy, same filter) per query",
        "same_set": sum(r["same_top3_set"] for r in ann_rows), "same_order": sum(r["same_top3_order"] for r in ann_rows),
        "total": len(ann_rows), "rows": ann_rows})
    lines = ["| Strategy | Split | Hit@1 | Evidence Hit@3 | MRR@3 | Gold page in top-3 | Split evidence suspected | Filter compliant |",
             "|---|---|---|---|---|---|---|---|"]
    for r in summary:
        fc = f"{r['filter_compliant']}/{r['filter_questions']}" if r["filter_questions"] else "N/A"
        lines.append(f"| {r['strategy']} | {r['split']} | {r['hit1']}/{r['n']} | {r['hit3']}/{r['n']} | {r['mrr3']} | "
                     f"{r['gold_page_in_top3']}/{r['n']} | {r['evidence_split_suspected']} | {fc} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"ANN check: {sum(r['same_top3_set'] for r in ann_rows)}/{len(ann_rows)} identical top-3 sets")


# ---------------------------------------------------------------- audit sheets
def coherence_positions(pages) -> list[tuple[int, int]]:
    """10 positions fixed before scoring: evenly spaced content pages, midpoint of page text."""
    content = sorted((p for p in pages if p.metadata["is_content_page"]), key=lambda p: (p.document_id, p.page))
    idx = [round(i * (len(content) - 1) / 9) for i in range(10)]
    return [(content[i].page, len(content[i].text) // 2) for i in idx]


def cmd_audit_sheets(_: argparse.Namespace) -> None:
    pages = _pages_from_results()
    by_page = {p.page: p for p in pages}
    audit = RESULTS / "audit"
    audit.mkdir(parents=True, exist_ok=True)

    chunks = {s: _read_jsonl(RESULTS / f"chunks_{s}.jsonl") for s in ("A", "B")}
    with (audit / "coherence_sheet.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["position_id", "page", "char_offset", "strategy", "chunk_id", "section_id", "chunk_text",
                    "label_0_1_2", "reason", "labeler"])
        for n, (page, offset) in enumerate(coherence_positions(pages), start=1):
            for strategy in ("A", "B"):
                cands = [c for c in chunks[strategy] if c["page"] == page and c["char_start"] <= offset < c["char_end"]]
                if not cands:   # offset fell in a dropped heading/whitespace gap: take the next chunk
                    cands = sorted((c for c in chunks[strategy] if c["page"] == page and c["char_start"] > offset),
                                   key=lambda c: c["char_start"])[:1]
                c = cands[0]
                w.writerow([f"pos{n:02d}", page, offset, strategy, c["chunk_id"], c["section_id"], c["text"], "", "", ""])

    traces = {s: _read_jsonl(RESULTS / "traces" / f"{s}.jsonl") for s in STRATEGIES}
    with (audit / "metadata_sheet.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "question_id", "question_filter", "rank", "chunk_id", "source", "page", "printed_page",
                    "chapter", "section_id", "auto_text_on_page", "auto_section_on_page", "auto_filter_ok",
                    "manual_complete_and_correct_0_1", "error_note", "labeler"])
        for s in STRATEGIES:
            for t in (t for t in traces[s] if t["split"] == "test"):
                for h in t["hits"]:
                    m = h["metadata"]
                    page = by_page[m["page"]]
                    on_page = page.text[m["char_start"]:m["char_end"]] == h["text"]
                    sec_ok = any(p.section_id == m["section_id"] for p in page.paragraphs
                                 if p.start <= m["char_start"] < p.end + 2)
                    fil = t["question_filter"]
                    fil_ok = "" if not fil else all(str(m.get(k)) == str(v) for k, v in fil.items())
                    w.writerow([s, t["question_id"], json.dumps(fil) if fil else "", h["rank"], h["chunk_id"],
                                m["source"], m["page"], m["printed_page"], m["chapter"], m["section_id"],
                                on_page, sec_ok, fil_ok, "", "", ""])

    with (audit / "grounding_sheet.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["strategy", "question_id", "question", "expected_answer", "gold_pages", "auto_hit_at_3",
                    "top1_text", "top2_text", "top3_text", "label_0_1_2", "reason", "labeler"])
        for s in STRATEGIES:
            for t in (t for t in traces[s] if t["split"] == "test"):
                texts = [f"[{h['chunk_id']} | p{h['metadata']['page']} | s={h['score']}]\n{h['text']}" for h in t["hits"]]
                w.writerow([s, t["question_id"], t["question"], t["expected_answer"],
                            ";".join(str(g["page"]) for g in t["gold"]), t["hit_at_3"], *texts, "", "", ""])
    print(f"Wrote audit sheets to {audit.relative_to(REPO)}")


def cmd_labels(_: argparse.Namespace) -> None:
    """Merge results/audit/manual_labels.json into labeled CSVs and aggregate the manual metrics."""
    audit = RESULTS / "audit"
    labels = json.loads((audit / "manual_labels.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {"status": labels["status"], "reviewed_by_student": labels["reviewed_by_student"],
                           "strategies": {}}

    def merge(sheet: str, key_col: str, section: str, target: str) -> list[dict[str, str]]:
        rows = list(csv.DictReader((audit / sheet).open(encoding="utf-8-sig")))
        for r in rows:
            strategy = r["strategy"] if section == "grounding" or r["strategy"] != "C" else "B"
            label, reason = labels[section][r[key_col]][strategy]
            r["label_0_1_2"], r["reason"], r["labeler"] = str(label), reason, labels["labeler"]
        with (audit / target).open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        return rows

    coh = merge("coherence_sheet.csv", "position_id", "coherence", "coherence_labeled.csv")
    gro = merge("grounding_sheet.csv", "question_id", "grounding", "grounding_labeled.csv")
    meta = list(csv.DictReader((audit / "metadata_sheet.csv").open(encoding="utf-8-sig")))
    error_ids = {e["chunk_id"] for e in labels["metadata"]["errors"]}
    traces = {s: _read_jsonl(RESULTS / "traces" / f"{s}.jsonl") for s in STRATEGIES}
    for s in STRATEGIES:
        c_scores = [int(r["label_0_1_2"]) for r in coh if r["strategy"] == ("B" if s == "C" else s)]
        g_scores = [int(r["label_0_1_2"]) for r in gro if r["strategy"] == s]
        m_rows = [r for r in meta if r["strategy"] == s]
        m_ok = sum(r["auto_text_on_page"] == "True" and r["auto_section_on_page"] == "True"
                   and r["chunk_id"] not in error_ids for r in m_rows)
        filt = {}
        for split in ("dev", "test"):
            fq = [t for t in traces[s] if t["split"] == split and t["question_filter"]]
            filt[split] = f"{sum(bool(t['top3_filter_compliant']) for t in fq)}/{len(fq)}" if fq else "N/A"
        out["strategies"][s] = {
            "coherence_mean": round(sum(c_scores) / len(c_scores), 2), "coherence_n": len(c_scores),
            "coherence_counts": {k: c_scores.count(k) for k in (0, 1, 2)},
            "metadata_complete_correct": f"{m_ok}/{len(m_rows)}",
            "filter_compliant": filt,
            "grounding_counts": {k: g_scores.count(k) for k in (0, 1, 2)},
            "grounding_mean": round(sum(g_scores) / len(g_scores), 2),
            "manual_hit3_test_equivalent_evidence": sum(1 for x in g_scores if x == 2),
        }
    _write_json(RESULTS / "manual_summary.json", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("ingest", "verify-benchmark", "reingest-check", "evaluate", "audit-sheets", "labels"):
        sub.add_parser(name)
    for name in ("build", "all"):
        p = sub.add_parser(name)
        p.add_argument("--reset", action="store_true", help="drop collections A/B before building")
    args = parser.parse_args()
    commands = {"ingest": cmd_ingest, "verify-benchmark": cmd_verify_benchmark, "build": cmd_build,
                "reingest-check": cmd_reingest_check, "evaluate": cmd_evaluate, "audit-sheets": cmd_audit_sheets,
                "labels": cmd_labels}
    if args.command == "all":
        for name in ("ingest", "verify-benchmark", "build", "reingest-check", "evaluate", "audit-sheets"):
            print(f"== {name}")
            commands[name](args)
    else:
        commands[args.command](args)


if __name__ == "__main__":
    main()
