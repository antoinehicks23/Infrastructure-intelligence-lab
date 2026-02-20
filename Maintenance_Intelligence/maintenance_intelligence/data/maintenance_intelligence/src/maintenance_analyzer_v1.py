import csv
from collections import Counter, defaultdict
from pathlib import Path

# -------- Config --------
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "engine_room_logs.csv"

KEYWORDS = [
    "vibration",
    "pressure",
    "temp",
    "temperature",
    "bearing",
    "leak",
    "trip",
    "cavitation",
    "contamination",
    "clog",
]


def load_logs(csv_path):
    """Load maintenance logs from a CSV into a list of dicts."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Normalize fields
            row["equipment"] = row["equipment"].strip()
            row["log_text"] = row["log_text"].strip()
            row["downtime_hours"] = float(row["downtime_hours"])
            rows.append(row)
    return rows


def keyword_frequency(logs, keywords):
    """Count keyword occurrences across all log_text fields."""
    counts = Counter()
    for row in logs:
        text = row["log_text"].lower()
        for kw in keywords:
            if kw in text:
                counts[kw] += 1
    return counts


def downtime_by_equipment(logs):
    """Sum downtime hours per equipment."""
    totals = defaultdict(float)
    for row in logs:
        totals[row["equipment"]] += row["downtime_hours"]
    return totals


def top_risk_equipment(downtime_totals, top_n=5):
    """Rank equipment by total downtime (descending)."""
    ranked = sorted(downtime_totals.items(), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: CSV not found at: {DATA_PATH}")
        return

    logs = load_logs(DATA_PATH)

    # 1) Keyword counts
    counts = keyword_frequency(logs, KEYWORDS)

    print("\nMost Frequent Failure Keywords:")
    for kw, n in counts.most_common():
        print(f"- {kw}: {n}")

    # 2) Downtime totals
    totals = downtime_by_equipment(logs)

    print("\nDowntime by Equipment:")
    for eq, hrs in sorted(totals.items(), key=lambda x: x[1], reverse=True):
        print(f"{eq}: {hrs:.2f} hours")

    # 3) Risk ranking
    ranked = top_risk_equipment(totals, top_n=5)

    print("\nHighest Risk Equipment (by downtime):")
    for eq, hrs in ranked:
        print(f"- {eq} ({hrs:.2f} hrs)")


if __name__ == "__main__":
    main()
