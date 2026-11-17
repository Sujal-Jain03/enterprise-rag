"""Run the Phase 8 evaluation harness and print + save the report.

Usage: python run_evaluation.py
Writes eval_report.json in the project root.
"""

from app.services.evaluation.evaluate import run_evaluation, save_report
from app.telemetry import setup_telemetry


def _fmt(value, ndigits=2):
    return f"{value:.{ndigits}f}" if value is not None else "n/a"


def main():
    mode = setup_telemetry()
    print(f"Telemetry mode: {mode}")
    print("Running evaluation (10 questions)...")

    report = run_evaluation()
    agg = report["aggregate"]

    print("\n--- Per-question ---")
    for r in report["results"]:
        if "recall_at_5" in r:
            faithful = r["faithful"]
            faithful_s = "YES" if faithful is True else ("NO" if faithful is False else "n/a")
            cites = r["citations"]
            cites_s = "OK" if (cites["all_valid"] and cites["has_citation"]) else "BAD"
            print(
                f"{r['id']} | found={r['found']} | "
                f"R@5={r['recall_at_5']:.0f} MRR={_fmt(r['mrr'])} nDCG={_fmt(r['ndcg_at_5'])} | "
                f"faithful={faithful_s} | cites {cites_s} | {r['latency_s']}s"
            )
        else:
            print(
                f"{r['id']} | found={r['found']} | abstained={r['abstained']} "
                f"(answer: {r['answer_abstained']}) | {r['latency_s']}s"
            )

    print("\n--- Aggregate ---")
    print(f"In-corpus: {report['n_in_corpus']} questions")
    print(f"  Recall@5: {_fmt(agg['recall_at_5'])} | MRR: {_fmt(agg['mrr'])} | nDCG@5: {_fmt(agg['ndcg_at_5'])}")
    print(f"  Faithful: {_fmt(agg['faithful_rate'])} ({agg['n_judged']} judged) | Answers question: {_fmt(agg['answers_rate'])}")
    print(f"  Citations valid: {_fmt(agg['citation_valid_rate'])} | Keyword hit (soft): {_fmt(agg['keyword_hit'])}")
    print(f"Out-of-corpus: {report['n_out_of_corpus']} questions | Abstention rate: {_fmt(agg['abstention_rate'])} (gate) / {_fmt(agg['answer_abstention_rate'])} (answer)")
    print(f"Latency: mean {_fmt(agg['latency_mean_s'])}s | median {_fmt(agg['latency_median_s'])}s | max {_fmt(agg['latency_max_s'])}s (first query includes one-time model load)")

    out = save_report(report, "eval_report.json")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
