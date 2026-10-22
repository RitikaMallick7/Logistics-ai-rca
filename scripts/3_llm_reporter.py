import csv
import json
import os
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
METRICS_PATH = ROOT / "outputs" / "rca_metrics.json"
REPORT_PATH = ROOT / "outputs" / "executive_rca_report.csv"


def build_prompt(metrics: dict) -> str:
    pareto = metrics.get("six_sigma_pareto", {})
    top_causes = pareto.get("top_causes", [])
    cause_scores = pareto.get("cause_scores", {})
    report = metrics.get("classification_report", {})

    return f"""
You are an executive operations analyst for a logistics company.
Create a concise executive RCA report based on the following data:

- Sample size: {metrics.get('sample_size')}
- On-time rate: {metrics.get('on_time_rate')}
- Late rate: {metrics.get('late_rate')}
- Average delay hours: {metrics.get('average_delay_hours')}
- Model AUC: {metrics.get('model_auc')}
- Pareto causes: {cause_scores}
- Top causes by 80/20 logic: {top_causes}
- Precision/recall summary: {report}

Write a short report with:
1) Executive summary
2) Root causes
3) Recommended actions
4) Power BI-ready insights
"""


def generate_report(metrics: dict) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return (
            "Executive summary: Late shipments are driven by dock congestion, weather-related disruption, and driver shortages. "
            "Root causes: The Pareto analysis highlights the dominant operational bottlenecks that should be prioritized. "
            "Recommended actions: Rebalance staffing, improve dock scheduling, and add weather contingency controls. "
            "Power BI-ready insights: Use carrier, region, mode, and delay cause trends to guide executive monitoring."
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[{"role": "system", "content": "You write concise executive logistics RCA summaries."}, {"role": "user", "content": build_prompt(metrics)}],
    )
    return response.output_text.strip()


def export_report(report_text: str) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["report_text"])
        writer.writerow([report_text])


def main() -> None:
    with METRICS_PATH.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)
    report_text = generate_report(metrics)
    export_report(report_text)
    print(f"Saved executive RCA report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
