"""Export collected survey results to CSV.

    python export_results.py results/survey.jsonl \
        --campaign campaigns/rajasthan_survey.json --out results/survey.csv

Columns are the call metadata (call_id, number, timestamp, status) followed by
one column per survey field. Field ids come from the campaign's `survey` block;
without --campaign, they're inferred from the answers seen in the results file.
"""

from __future__ import annotations

import argparse
import sys

from voiceos.survey.definition import SurveyDefinition
from voiceos.survey.store import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(prog="voiceos-export-results")
    parser.add_argument("results", help="results JSONL written by serve_telephony.py")
    parser.add_argument("--campaign", help="campaign JSON (for the field order)")
    parser.add_argument("--out", default="results/survey.csv")
    args = parser.parse_args()

    store = ResultStore(args.results)
    records = store.records()
    if not records:
        sys.exit(f"no results in {args.results}")

    if args.campaign:
        survey = SurveyDefinition.from_campaign_file(args.campaign)
        field_ids = survey.field_ids if survey else []
    else:
        # Infer the union of answer keys across all records, order-preserving.
        field_ids = list(dict.fromkeys(
            k for r in records for k in r.get("answers", {})
        ))

    n = store.export_csv(args.out, field_ids)
    print(f"wrote {n} rows and {len(field_ids)} survey columns to {args.out}")


if __name__ == "__main__":
    main()
