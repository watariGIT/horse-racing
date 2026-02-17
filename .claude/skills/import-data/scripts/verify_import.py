"""Verify BigQuery import by checking row counts for all tables."""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BigQuery import row counts")
    parser.add_argument(
        "--env",
        choices=["prod", "dev"],
        default="prod",
        help="Environment to check (default: prod)",
    )
    parser.add_argument(
        "--project",
        default="horse-racing-ml-dev",
        help="GCP project ID (default: horse-racing-ml-dev)",
    )
    args = parser.parse_args()

    dataset = "horse_racing" if args.env == "prod" else "horse_racing_dev"
    tables = ["races_raw", "horse_results_raw", "jockey_results_raw"]

    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=args.project)
    except Exception as e:
        print(f"Error: Failed to create BigQuery client: {e}")
        sys.exit(1)

    print(f"Checking {args.env} environment (dataset: {dataset})")
    print("-" * 50)

    all_ok = True
    for table in tables:
        try:
            query = f"SELECT COUNT(*) as cnt FROM `{dataset}.{table}`"
            result = client.query(query).result()
            for row in result:
                count = row.cnt
                status = "OK" if count > 0 else "EMPTY"
                print(f"  {table}: {count:,} rows [{status}]")
                if count == 0:
                    all_ok = False
        except Exception as e:
            print(f"  {table}: ERROR - {e}")
            all_ok = False

    print("-" * 50)
    if all_ok:
        print("All tables have data.")
    else:
        print("Some tables are empty or had errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
