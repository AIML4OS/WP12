import json
import sys
import argparse
from pathlib import Path

from pipeline.report_pipeline import UpdatedAnnualReportPipeline
#from config.settings import Config


def main():
    API_KEY = "API_KEY"
    BASE_URL = "https://llm.lab.sspcloud.fr/api"
    MODEL_ID = "gpt-oss:120b"

    pipeline = UpdatedAnnualReportPipeline(API_KEY, BASE_URL, MODEL_ID)

    reports = ["pdf/test.pdf", "pdf/Ryanair Holdings PLC (1).pdf"]

    results = []
    for report_path in reports:
        try:
            result = pipeline.process_report(report_path)
            results.append(result)

            print(f"Status: {result['status']}")
            if result['status'] != 'failed':
                extracted = result["extracted_data"]
                if "error" not in extracted:
                    print(f"Company: {extracted.get('company_name', 'Unknown')}")
                    print(f"Year: {extracted.get('report_year', 'Unknown')}")
                    print(f"Revenue: €{extracted.get('financial_metrics', {}).get('total_revenue', 'N/A')}M")
                    print(f"Confidence: {extracted.get('extraction_confidence', 'N/A')}/10")
                print(json.dumps(result["extracted_data"], indent=2))
            print("-" * 60)

        except Exception as e:
            print(f"Error processing {report_path}: {e}")

    # Save detailed results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Processed {len(results)} reports.")

if __name__ == "__main__":
    main()
