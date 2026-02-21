"""
SafeStride — Step 4, Part 1: Edge Scoring Runner
=================================================
Standalone script that runs the full CrimeScorer pipeline and prints a
formatted summary report based on the output CSV.

Usage (always use the project venv):
    Windows:  venv\\Scripts\\activate  then  python scripts/score_edges.py
    Linux/Mac: source venv/bin/activate  then  python scripts/score_edges.py
"""

# ── Venv guard — catch missing dependencies early with a clear message ──────
try:
    import geopandas  # noqa: F401
except ModuleNotFoundError:
    print(
        "\n❌  Required packages not found.\n"
        "   You must run this script inside the project virtual environment.\n\n"
        "   Activate it first:\n"
        "     Windows:   venv\\Scripts\\activate\n"
        "     Linux/Mac: source venv/bin/activate\n\n"
        "   Then re-run:\n"
        "     python scripts/score_edges.py\n"
    )
    raise SystemExit(1)

import logging
import sys
from pathlib import Path

import pandas as pd

# ── Make sure the project root is on sys.path ──────────────────────────────
# This lets us import from src/ regardless of where the script is run from.
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.safety.crime_scorer import CrimeScorer  # noqa: E402


# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────
# Configure the root "safestride" logger so all pipeline logs go to stdout
# with timestamps. This gives visibility into each pipeline stage.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────
def run_pipeline() -> Path:
    """
    Instantiate CrimeScorer and run the full pipeline.

    Returns
    -------
    Path
        Absolute path to the saved edge_crime_scores.csv.
    """
    scorer = CrimeScorer()
    scorer.run()

    output_path = PROJECT_ROOT / "data" / "processed" / "edge_crime_scores.csv"
    return output_path


# ──────────────────────────────────────────────
# SUMMARY REPORT
# ──────────────────────────────────────────────
def print_report(csv_path: Path) -> None:
    """
    Load the saved scores CSV and print a formatted summary report.

    Parameters
    ----------
    csv_path : Path
        Path to edge_crime_scores.csv produced by CrimeScorer.save_scores().
    """
    if not csv_path.exists():
        print(f"\n❌  Output file not found at: {csv_path}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    scores = df["crime_risk_score"]

    total_edges   = len(df)
    with_crime    = (scores > 0).sum()
    without_crime = (scores == 0).sum()

    sep = "=" * 60

    print(f"\n{sep}")
    print("  SAFESTRIDE — CRIME SCORING COMPLETE")
    print(sep)

    print(f"  Total edges scored:        {total_edges:>10,}")
    print(f"  Edges with crime data:     {with_crime:>10,}  ({with_crime / total_edges * 100:.1f}%)")
    print(f"  Edges with zero crimes:    {without_crime:>10,}  ({without_crime / total_edges * 100:.1f}%)")

    print(f"\n  Score Distribution:")
    print(f"    Min:    {scores.min():.4f}")
    print(f"    Max:    {scores.max():.4f}")
    print(f"    Mean:   {scores.mean():.4f}")
    print(f"    Median: {scores.median():.4f}")
    print(f"    Std:    {scores.std():.4f}")

    print(f"\n  Top 5 Most Dangerous Edges:")
    print(f"    {'(u, v, key)':<40}  score")
    print(f"    {'─' * 50}")
    top5 = df.nlargest(5, "crime_risk_score")
    for _, row in top5.iterrows():
        label = f"({int(row['u'])}, {int(row['v'])}, {int(row['key'])})"
        print(f"    {label:<40}  {row['crime_risk_score']:.4f}")

    print(f"\n  Output: {csv_path.resolve()}")
    print(f"{sep}\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    output_csv = run_pipeline()
    print_report(output_csv)
