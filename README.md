# Logistics AI RCA & Executive Insights System

This project creates a lightweight end-to-end analytics workflow for logistics shipment performance.

## What it does
- Generates a synthetic TMS shipment dataset in data/tms_shipments.csv
- Trains an XGBoost classifier to predict on-time delivery and estimates model performance
- Applies Six Sigma-style Pareto logic to identify dominant delay drivers
- Produces an executive RCA report as a CSV file for Power BI consumption

## Files
- data/tms_shipments.csv: Generated synthetic shipment data
- outputs/rca_metrics.json: ML and Pareto analytics output
- outputs/executive_rca_report.csv: Final executive summary for dashboards

## Run the workflow
```bash
pip install -r requirements.txt
python scripts/1_generate_data.py
python scripts/2_ml_six_sigma.py
python scripts/3_llm_reporter.py
```

## Notes
- The LLM reporter uses OpenAI when OPENAI_API_KEY is set.
- If no API key is present, it falls back to a deterministic executive summary.
