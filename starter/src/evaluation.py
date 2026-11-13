"""Automatic retrieval metrics (Hit@1, Evidence Hit@3) and per-query traces.

A retrieved chunk is a hit when it comes from a gold (source, page) AND contains
the full gold evidence quote (after the same normalization as ingestion, case-
and whitespace-insensitive). Alternative gold evidence listed in the benchmark
is accepted. Coherence and grounding are NOT inferred here: they are manual.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .ingestion import normalize_text


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", normalize_text(text)).strip().lower()


def gold_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    options = [{"source_id": question["gold_source_id"], "page": question["gold_page"],
                "evidence_quote": question["evidence_quote"]}]
    for alt in question.get("alternative_gold", []):
        options.append({"source_id": alt["gold_source_id"], "page": alt["gold_page"],
                        "evidence_quote": alt["evidence_quote"]})
    return options


def _partial(chunk_text: str, quote: str) -> bool:
    """True if a sizeable piece of the quote is present (evidence split across chunks)."""
    words = _norm(quote).split()
    if len(words) < 6:
        return False
    half = max(4, len(words) // 2)
    text = _norm(chunk_text)
    return " ".join(words[:half]) in text or " ".join(words[-half:]) in text


def judge_hit(question: dict[str, Any], hit: dict[str, Any]) -> dict[str, Any]:
    meta = hit["metadata"]
    for opt in gold_options(question):
        same_place = meta.get("document_id") == opt["source_id"] and int(meta.get("page")) == int(opt["page"])
        if same_place and _norm(opt["evidence_quote"]) in _norm(hit["text"]):
            return {"is_hit": True, "gold_page_match": True, "evidence_partial": False}
    page_match = any(meta.get("document_id") == o["source_id"] and int(meta.get("page")) == int(o["page"])
                     for o in gold_options(question))
    partial = any(_partial(hit["text"], o["evidence_quote"]) for o in gold_options(question))
    return {"is_hit": False, "gold_page_match": page_match, "evidence_partial": partial}


def filter_compliant(hits: list[dict[str, Any]], filters: dict[str, Any] | None) -> bool | None:
    if not filters:
        return None
    return all(str(h["metadata"].get(k)) == str(v) for h in hits for k, v in filters.items())


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]], strategy: str,
                   applied_filter: dict[str, Any] | None) -> dict[str, Any]:
    judged = []
    for hit in hits:
        judged.append({**hit, **judge_hit(question, hit)})
    first_hit = next((h["rank"] for h in judged if h["is_hit"]), None)
    return {
        "strategy": strategy,
        "question_id": question["id"],
        "split": question["split"],
        "type": question.get("type"),
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold": gold_options(question),
        "question_filter": question.get("filter"),
        "applied_filter": applied_filter,
        "hit_at_1": bool(judged and judged[0]["is_hit"]),
        "hit_at_3": first_hit is not None and first_hit <= 3,
        "first_hit_rank": first_hit,
        "gold_page_in_top3": any(h["gold_page_match"] for h in judged),
        "evidence_split_suspected": first_hit is None and any(h["evidence_partial"] for h in judged),
        "top3_filter_compliant": filter_compliant(judged, question.get("filter")),
        "hits": judged,
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for t in traces:
        groups[(t["strategy"], t["split"])].append(t)
        groups[(t["strategy"], "all")].append(t)
    rows = []
    for (strategy, split), items in sorted(groups.items(), key=lambda kv: (kv[0][0], ["dev", "test", "all"].index(kv[0][1]))):
        with_filter = [t for t in items if t["question_filter"]]
        rows.append({
            "strategy": strategy, "split": split, "n": len(items),
            "hit1": sum(t["hit_at_1"] for t in items),
            "hit3": sum(t["hit_at_3"] for t in items),
            "mrr3": round(sum(1 / t["first_hit_rank"] for t in items if t["first_hit_rank"]) / len(items), 4),
            "gold_page_in_top3": sum(t["gold_page_in_top3"] for t in items),
            "evidence_split_suspected": sum(t["evidence_split_suspected"] for t in items),
            "filter_questions": len(with_filter),
            "filter_compliant": sum(bool(t["top3_filter_compliant"]) for t in with_filter),
        })
    return rows
