"""Evaluation runner: runs the gold eval set through generate() and scores it.

Per-question results:
- in-corpus: Recall@5 / MRR / nDCG@5 (vs expected source), LLM-judge
  faithfulness + answer relevance, citation validity, keyword hit (soft)
- out-of-corpus: abstention — found must be False

Aggregates are means over the in-corpus questions (retrieval + judge) and
the abstention rate over the out-of-corpus questions. Latency is reported
as mean / median / max — the first query includes one-time BGE + FlashRank
model loads, so the max is expected to be much larger than the median.
"""

import json
import re
import statistics
import time
from pathlib import Path

import logfire

from app.services.evaluation import metrics
from app.services.evaluation.judge import judge_answer
from app.services.generation.generator import generate

EVAL_SET_PATH = Path(__file__).parent / "eval_set.json"
CITATION_RE = re.compile(r"\[(\d+)\]")


def load_eval_set() -> dict:
    with open(EVAL_SET_PATH, encoding="utf-8") as f:
        return json.load(f)


def _check_citations(answer: str, num_chunks: int) -> dict:
    """Every [n] must satisfy 1 <= n <= num_chunks; at least one citation expected."""
    cited = [int(n) for n in CITATION_RE.findall(answer)]
    return {
        "cited": cited,
        "has_citation": len(cited) > 0,
        "all_valid": all(1 <= n <= num_chunks for n in cited),
    }


def evaluate_question(q: dict) -> dict:
    """Run one eval question through generate() and score the result."""
    t0 = time.perf_counter()
    result = generate(q["question"])
    latency = time.perf_counter() - t0

    sources = result.get("sources") or []
    filenames = [s.get("filename", "") for s in sources]
    answer = result.get("answer", "")

    row = {
        "id": q["id"],
        "question": q["question"],
        "found": result.get("found"),
        "error": result.get("error"),
        "latency_s": round(latency, 2),
        "retrieved_filenames": filenames,
        "top_vector_score": result.get("top_vector_score"),
        "answer": answer,
    }

    if q["in_corpus"]:
        expected = q["expected_source"]
        row["recall_at_5"] = metrics.recall_at_k(filenames, expected)
        row["mrr"] = metrics.reciprocal_rank(filenames, expected)
        row["ndcg_at_5"] = metrics.ndcg_at_k(filenames, expected)
        keywords = q.get("expected_keywords") or []
        row["keyword_hit"] = (
            sum(1 for kw in keywords if kw.lower() in answer.lower()) / len(keywords)
            if keywords
            else None
        )
        if result.get("found"):
            context_texts = [s.get("text", "") for s in sources]
            verdict = judge_answer(q["question"], answer, context_texts)
            row["faithful"] = verdict["faithful"]
            row["answers_question"] = verdict["answers_question"]
            row["judge_reason"] = verdict["reason"]
            row["judge_error"] = verdict["error"]
            row["citations"] = _check_citations(answer, len(sources))
        else:
            row["faithful"] = None
            row["answers_question"] = None
            row["judge_reason"] = ""
            row["judge_error"] = None
            row["citations"] = {"cited": [], "has_citation": False, "all_valid": True}
    else:
        row["abstained"] = result.get("found") is False
        # Answer-level abstention: the LLM may refuse even when the gate
        # passed a barely-relevant chunk (eval run 2, q10).
        row["answer_abstained"] = result.get("found") is False or answer.strip().lower().startswith("i don't know")

    return row


def _mean(values):
    return round(sum(values) / len(values), 4) if values else None


@logfire.instrument("run_evaluation")
def run_evaluation() -> dict:
    """Run the full eval set and return the report dict."""
    eval_set = load_eval_set()
    rows = []
    for i, q in enumerate(eval_set["questions"]):
        if i > 0:
            time.sleep(2)  # pace Groq calls to avoid 429 rate limits
        rows.append(evaluate_question(q))

    in_corpus = [r for r, q in zip(rows, eval_set["questions"]) if q["in_corpus"]]
    out_of_corpus = [r for r, q in zip(rows, eval_set["questions"]) if not q["in_corpus"]]
    judged = [r for r in in_corpus if r["faithful"] is not None]

    latencies = sorted(r["latency_s"] for r in rows)
    report = {
        "eval_set_version": eval_set.get("version"),
        "n_questions": len(rows),
        "n_in_corpus": len(in_corpus),
        "n_out_of_corpus": len(out_of_corpus),
        "aggregate": {
            "recall_at_5": _mean([r["recall_at_5"] for r in in_corpus]),
            "mrr": _mean([r["mrr"] for r in in_corpus]),
            "ndcg_at_5": _mean([r["ndcg_at_5"] for r in in_corpus]),
            "keyword_hit": _mean(
                [r["keyword_hit"] for r in in_corpus if r["keyword_hit"] is not None]
            ),
            "faithful_rate": _mean([1.0 if r["faithful"] else 0.0 for r in judged]),
            "answers_rate": _mean(
                [1.0 if r["answers_question"] else 0.0 for r in judged]
            ),
            "n_judged": len(judged),
            "citation_valid_rate": _mean(
                [
                    1.0 if r["citations"]["all_valid"] and r["citations"]["has_citation"] else 0.0
                    for r in in_corpus
                    if r["found"]
                ]
            ),
            "abstention_rate": _mean(
                [1.0 if r["abstained"] else 0.0 for r in out_of_corpus]
            ),
            "answer_abstention_rate": _mean(
                [1.0 if r["answer_abstained"] else 0.0 for r in out_of_corpus]
            ),
            "latency_mean_s": _mean(latencies),
            "latency_median_s": round(statistics.median(latencies), 2) if latencies else None,
            "latency_max_s": latencies[-1] if latencies else None,
        },
        "results": rows,
    }
    return report


def save_report(report: dict, path) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
