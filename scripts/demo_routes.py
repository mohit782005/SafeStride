"""
SafeStride — Step 4, Part 5: End-to-End Routing Demo
=====================================================
Standalone script that runs the full pipeline end to end:

  1. CrimeScorer().run()         → crime-scored NetworkX graph
  2. SafeStrideRouter            → set_edge_weights()
  3. find_pareto_routes()        → 3 Pareto-optimal routes
  4. Print a clean comparison report to stdout

Usage (always activate the project venv first):
    Windows:   venv\\Scripts\\activate   then   python scripts/demo_routes.py
    Linux/Mac: source venv/bin/activate  then   python scripts/demo_routes.py
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
        "     python scripts/demo_routes.py\n"
    )
    raise SystemExit(1)

import logging
import sys
from pathlib import Path

# ── Make sure the project root is on sys.path ──────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.safety.crime_scorer import CrimeScorer          # noqa: E402
from src.routing.router import SafeStrideRouter           # noqa: E402


# ──────────────────────────────────────────────
# LOGGING SETUP
# ──────────────────────────────────────────────
# Keep pipeline logs visible but below the final report so they don't
# drown out the formatted output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


# ──────────────────────────────────────────────
# FIXED DEMO COORDINATES (Chicago downtown)
# ──────────────────────────────────────────────
ORIGIN_LAT = 41.8827
ORIGIN_LON = -87.6233
DEST_LAT   = 41.8750
DEST_LON   = -87.6350


# ──────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────
def run_pipeline() -> list[dict]:
    """
    Run the full SafeStride pipeline and return the three Pareto routes.

    Steps
    -----
    1. Build or load the crime-scored graph via CrimeScorer.run().
    2. Wrap it in SafeStrideRouter.
    3. Call find_pareto_routes() — internally re-weights edges three times
       and runs Dijkstra for each scenario.

    Returns
    -------
    list[dict]
        The three route dicts (safety / balanced / speed) each containing:
        label, alpha, beta, route, total_distance_m, avg_safety_score,
        max_danger_score, estimated_time_min.
    """
    print("\nRunning CrimeScorer pipeline …")
    scorer = CrimeScorer()
    graph  = scorer.run()

    print("Initialising SafeStrideRouter …")
    router = SafeStrideRouter(graph)

    # set_edge_weights() is called internally by find_pareto_routes() for each
    # preset, so we don't need a standalone call here.
    print("Computing Pareto routes …\n")
    routes = router.find_pareto_routes(
        origin_lat=ORIGIN_LAT,
        origin_lon=ORIGIN_LON,
        dest_lat=DEST_LAT,
        dest_lon=DEST_LON,
    )
    return routes


# ──────────────────────────────────────────────
# REPORT PRINTER
# ──────────────────────────────────────────────
_ICONS = {
    "safety_maximized": "🛡️ ",
    "balanced":         "⚖️ ",
    "speed_maximized":  "⚡",
}

_TITLES = {
    "safety_maximized": "SAFETY ROUTE",
    "balanced":         "BALANCED ROUTE",
    "speed_maximized":  "SPEED ROUTE",
}

SEP = "=" * 60


def print_report(routes: list[dict]) -> None:
    """
    Print the formatted route comparison report to stdout.

    Parameters
    ----------
    routes : list[dict]
        Three Pareto route dicts as returned by find_pareto_routes().
    """
    print(SEP)
    print("SAFESTRIDE — ROUTE OPTIONS")
    print(f"Origin:      {ORIGIN_LAT}, {ORIGIN_LON}")
    print(f"Destination: {DEST_LAT}, {DEST_LON}")
    print(SEP)

    for r in routes:
        label  = r["label"]
        icon   = _ICONS.get(label, "•")
        title  = _TITLES.get(label, label.upper())

        dist_m   = r["total_distance_m"]
        time_min = r["estimated_time_min"]
        avg_d    = r["avg_safety_score"]
        max_d    = r["max_danger_score"]

        print(f"\n{icon}  {title}")
        print(f"   Distance:       {dist_m:>8,.0f} m")
        print(f"   Est. Time:      {time_min:>7.1f} min")
        print(f"   Avg Danger:     {avg_d:>8.3f}")
        print(f"   Max Danger:     {max_d:>8.3f}")

    print(f"\n{SEP}")
    print("Recommendation: SAFETY route avoids the most dangerous segments.")
    print(SEP)
    print()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    routes = run_pipeline()
    print_report(routes)
