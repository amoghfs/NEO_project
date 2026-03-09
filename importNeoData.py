import argparse
from datetime import datetime, timedelta, timezone

from neo_pipeline import fetch_neos, init_db, upsert_raw


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch NEO data from NASA and store it in neo.db")
    parser.add_argument(
        "--days-back",
        type=int,
        default=1,
        help="How many days back from today to include in the feed window",
    )
    parser.add_argument(
        "--days-forward",
        type=int,
        default=5,
        help="How many days forward from today to include in the feed window",
    )
    args = parser.parse_args()

    if args.days_back < 0 or args.days_forward < 0:
        raise ValueError("days-back and days-forward must be non-negative")
    if args.days_back + args.days_forward + 1 > 7:
        raise ValueError("NASA feed supports a maximum 7-day inclusive window")

    init_db()

    today_utc = datetime.now(timezone.utc).date()
    start_date = (today_utc - timedelta(days=args.days_back)).isoformat()
    end_date = (today_utc + timedelta(days=args.days_forward)).isoformat()

    rows = fetch_neos(start_date, end_date)
    upserted = upsert_raw(rows)

    print(f"Fetched {len(rows)} NEO rows from NASA for {start_date} to {end_date}.")
    print(f"Upserted {upserted} rows into near_earth_objects.")


if __name__ == "__main__":
    main()
