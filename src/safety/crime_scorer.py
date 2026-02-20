"""
SafeStride — Crime Scorer (Step 3, Parts 1-2)
=====================================================
Loads the Chicago road network and geocoded crime data, preparing
both for spatial join and edge-level safety scoring in later parts.

Why EPSG:3435?
  WGS-84 (EPSG:4326) uses degrees — distances are meaningless.
  EPSG:3435 is the Illinois State Plane East (US feet) projection,
  which preserves distances and areas for Chicago. This is critical
  for accurate buffer-based spatial joins when we assign crimes to
  nearby road segments.
"""

import gc
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.strtree import STRtree

# ──────────────────────────────────────────────
# LOGGER
# ──────────────────────────────────────────────
# Module-level logger under the safestride namespace.
# Downstream code can configure the root "safestride" logger
# to control all SafeStride log output centrally.
logger = logging.getLogger("safestride.crime_scorer")


class CrimeScorer:
    """
    Loads and prepares the two core datasets for safety scoring:
      1. Chicago walk network (graph + edge GeoDataFrame)
      2. Cleaned crime incidents (point GeoDataFrame)

    Both are projected to a common CRS so spatial operations
    (buffers, nearest-edge lookups) use real-world distances.
    """

    def __init__(self, graph_crs: str = "EPSG:3435") -> None:
        """
        Initialise the scorer with a target CRS.

        Parameters
        ----------
        graph_crs : str
            Coordinate Reference System to project both datasets into.
            Default is EPSG:3435 (Illinois State Plane East, US feet).
        """
        self.graph_crs = graph_crs

        # These will be populated by load_graph() and load_crimes().
        # Keeping them as None lets callers check readiness before scoring.
        self.graph = None          # networkx.MultiDiGraph (projected)
        self.edges_gdf = None      # GeoDataFrame of edges with geometry
        self.crimes_gdf = None     # GeoDataFrame of crime points
        self.spatial_index = None  # STRtree for fast nearest-edge lookups
        self.snapped = None        # DataFrame mapping crimes → edges

    # ──────────────────────────────────────────
    # GRAPH LOADING
    # ──────────────────────────────────────────
    def load_graph(self) -> None:
        """
        Download Chicago's walkable street network via OSMnx,
        project it to the target CRS, and extract the edge GeoDataFrame.

        Why "walk" network?
          SafeStride recommends *walkable* routes, so we need sidewalks
          and pedestrian paths — not just driveable roads. OSMnx's
          network_type="walk" includes footways, paths, and residential
          streets that pedestrians actually use.

        Why extract edges?
          The edge GeoDataFrame contains the LineString geometry for
          every road segment. We'll use these geometries to spatially
          join crime points to their nearest edge in later parts.
        """
        logger.info("Downloading Chicago walk network from OSM...")
        graph = ox.graph_from_place(
            "Chicago, Illinois, USA",
            network_type="walk",
        )

        # Project from WGS-84 (lat/lon) → EPSG:3435 (Illinois feet).
        # This makes .buffer(distance) use US feet, not degrees.
        logger.info("Projecting graph to %s...", self.graph_crs)
        self.graph = ox.project_graph(graph, to_crs=self.graph_crs)

        # Extract edges as a GeoDataFrame for spatial operations.
        # ox.graph_to_gdfs returns (nodes_gdf, edges_gdf).
        # We only need edges for scoring; nodes are in the graph object.
        _, self.edges_gdf = ox.graph_to_gdfs(self.graph)

        # Free the unprojected graph to save memory — it can be large.
        del graph
        gc.collect()

        logger.info(
            "Graph loaded: %d nodes, %d edges",
            len(self.graph.nodes),
            len(self.edges_gdf),
        )

    # ──────────────────────────────────────────
    # CRIME DATA LOADING
    # ──────────────────────────────────────────
    def load_crimes(self, filepath: str = "data/crimes_cleaned.csv") -> None:
        """
        Load the cleaned crime CSV and convert it to a projected GeoDataFrame.

        Parameters
        ----------
        filepath : str
            Path to the cleaned crime CSV (relative to project root or absolute).

        Raises
        ------
        FileNotFoundError
            If the CSV doesn't exist — with a helpful message pointing
            the user to the fetch script.

        Why convert to GeoDataFrame?
          Raw CSV has lat/lon as plain floats. A GeoDataFrame stores them
          as Shapely Point geometries, enabling spatial operations like
          .sjoin_nearest() and .buffer() that we'll need for scoring.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(
                f"Crime data not found at '{path.resolve()}'.\n"
                f"Run 'python scripts/fetch_crime_data.py' first to download it."
            )

        logger.info("Loading crime data from %s...", path)
        df = pd.read_csv(path, parse_dates=["date"])

        # Build Point geometries from longitude (x) and latitude (y).
        # Note: GeoDataFrame expects (x, y) = (lon, lat), not (lat, lon).
        self.crimes_gdf = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
            crs="EPSG:4326",  # Raw data is in WGS-84 (standard GPS coords)
        )

        # Project to the same CRS as the road network so spatial joins
        # use consistent distance units (US feet in EPSG:3435).
        self.crimes_gdf = self.crimes_gdf.to_crs(self.graph_crs)

        logger.info(
            "Crime data loaded: %d records, CRS=%s",
            len(self.crimes_gdf),
            self.crimes_gdf.crs,
        )

    # ──────────────────────────────────────────
    # SPATIAL INDEX
    # ──────────────────────────────────────────
    def build_spatial_index(self) -> None:
        """
        Build an STRtree spatial index over the edge geometries.

        Why STRtree?
          A brute-force nearest-edge search is O(crimes × edges) — with
          ~200k crimes and ~100k+ edges that's 20 billion distance calcs.
          STRtree (Sort-Tile-Recursive tree) is a read-optimised R-tree
          that answers nearest-neighbor queries in O(log N), reducing
          the total cost to O(crimes × log(edges)).

        Why build it separately?
          The tree is immutable once built. Separating construction from
          querying lets us build once and snap multiple datasets (e.g.
          crimes from different years) without rebuilding.
        """
        if self.edges_gdf is None:
            raise RuntimeError(
                "Edges not loaded. Call load_graph() before build_spatial_index()."
            )

        # STRtree indexes the geometry column of the edges GeoDataFrame.
        # Each geometry is a LineString representing one road segment.
        self.spatial_index = STRtree(self.edges_gdf.geometry.values)

        logger.info(
            "Spatial index built over %d edge geometries.",
            len(self.edges_gdf),
        )

    # ──────────────────────────────────────────
    # SNAP CRIMES TO EDGES
    # ──────────────────────────────────────────
    def snap_crimes_to_edges(self, buffer_feet: float = 262.5) -> None:
        """
        For each crime point, find the nearest road edge within a distance
        threshold using the spatial index. Results are stored in self.snapped.

        Parameters
        ----------
        buffer_feet : float
            Maximum snap distance in US survey feet (EPSG:3435 units).
            Default is 262.5 ft ≈ 80 meters. Crimes farther than this
            from any road edge are discarded — they likely occurred in
            parks, buildings, or other off-network locations.

        Why not meters?
          EPSG:3435 uses US survey feet, so all distances in this CRS
          are in feet. 80 meters ≈ 262.5 feet. Using the native CRS
          unit avoids an extra conversion and keeps the code honest
          about what the numbers mean.

        Result
        ------
        self.snapped : pd.DataFrame
            Columns: crime_idx, u, v, key
            - crime_idx: index into self.crimes_gdf
            - u, v, key: the (u, v, key) multi-index of the matched edge
              in self.edges_gdf / self.graph
        """
        if self.spatial_index is None:
            raise RuntimeError(
                "Spatial index not built. Call build_spatial_index() first."
            )
        if self.crimes_gdf is None:
            raise RuntimeError(
                "Crime data not loaded. Call load_crimes() first."
            )

        logger.info(
            "Snapping %d crimes to nearest edge (max %d ft / ~80 m)...",
            len(self.crimes_gdf),
            int(buffer_feet),
        )

        crime_geoms = self.crimes_gdf.geometry.values

        # STRtree.nearest returns the index (positional) of the nearest
        # geometry in the tree for EACH query geometry — fully vectorized.
        # This is a single bulk call, not a Python loop.
        nearest_idx = self.spatial_index.nearest(crime_geoms)

        # Compute the actual distance from each crime to its nearest edge.
        # We need this to enforce the buffer threshold — STRtree.nearest
        # always returns *something*, even if it's kilometres away.
        nearest_edge_geoms = self.edges_gdf.geometry.values[nearest_idx]
        distances = np.array([
            crime.distance(edge)
            for crime, edge in zip(crime_geoms, nearest_edge_geoms)
        ])

        # Keep only crimes within the buffer distance.
        within_mask = distances <= buffer_feet

        # Extract the (u, v, key) multi-index of matched edges.
        # edges_gdf has a MultiIndex of (u, v, key) — these are the
        # OSMnx node IDs that identify each edge in the NetworkX graph.
        matched_edge_indices = self.edges_gdf.index[nearest_idx[within_mask]]

        self.snapped = pd.DataFrame({
            "crime_idx": self.crimes_gdf.index[within_mask],
            "u": matched_edge_indices.get_level_values(0),
            "v": matched_edge_indices.get_level_values(1),
            "key": matched_edge_indices.get_level_values(2),
        }).reset_index(drop=True)

        snapped_count = len(self.snapped)
        discarded_count = len(self.crimes_gdf) - snapped_count

        logger.info(
            "Snapped %d crimes to edges (%.1f%%). Discarded %d beyond %d ft.",
            snapped_count,
            snapped_count / len(self.crimes_gdf) * 100,
            discarded_count,
            int(buffer_feet),
        )

    # ──────────────────────────────────────────
    # CRIME WEIGHTS
    # ──────────────────────────────────────────

    # Severity weights by crime type.
    # Higher = more dangerous for pedestrians. These are based on the FBI
    # Uniform Crime Reporting hierarchy, adjusted for pedestrian risk:
    #   - Violent person-to-person crimes rank highest (HOMICIDE, ASSAULT)
    #   - Property crimes rank lower (THEFT, CRIMINAL DAMAGE)
    #   - Unknown/other types get a baseline of 1.0
    CRIME_TYPE_WEIGHTS: dict[str, float] = {
        "HOMICIDE":              10.0,
        "CRIM SEXUAL ASSAULT":    9.0,
        "HUMAN TRAFFICKING":      9.0,
        "KIDNAPPING":             8.0,
        "STALKING":               7.5,
        "ROBBERY":                7.0,
        "ASSAULT":                6.5,
        "BATTERY":                5.0,
        "BURGLARY":               4.0,
        "MOTOR VEHICLE THEFT":    3.5,
        "THEFT":                  3.0,
        "NARCOTICS":              2.5,
        "CRIMINAL DAMAGE":        2.0,
    }
    _DEFAULT_CRIME_WEIGHT = 1.0

    def compute_crime_weights(self) -> None:
        """
        Compute a composite danger weight for each snapped crime incident.

        The weight is the product of three independent factors:

        1. **Crime type severity** — a static lookup based on `primary_type`.
           Violent crimes (HOMICIDE=10) outweigh property crimes (THEFT=3).

        2. **Temporal multiplier** — crimes at night are scarier for pedestrians.
             Hours 22–23, 0–4  → ×1.8  (late night / early morning)
             Hours 18–21       → ×1.3  (evening)
             Otherwise         → ×1.0  (daytime)

        3. **Day-of-week multiplier** — weekend nights tend to be rowdier.
             Friday (4), Saturday (5) → ×1.2
             Sunday (6)               → ×1.1
             Otherwise                → ×1.0

        Why multiply?
          Multiplication means the factors *compound*: a ROBBERY (7.0)
          at 1 AM (×1.8) on a Saturday (×1.2) scores 7.0 × 1.8 × 1.2 = 15.12,
          while the same robbery at noon on a Tuesday scores just 7.0.
          This matches intuition — the same crime is more threatening
          in dark, less-populated conditions.

        Result
        ------
        Adds a `final_weight` column to self.snapped.
        """
        if self.snapped is None or self.snapped.empty:
            raise RuntimeError(
                "No snapped crimes. Call snap_crimes_to_edges() first."
            )

        logger.info("Computing crime weights for %d snapped incidents...", len(self.snapped))

        # Pull hour and day_of_week from the original crimes GeoDataFrame
        # using the crime_idx stored during snapping.
        crime_data = self.crimes_gdf.loc[self.snapped["crime_idx"]]

        # ── 1. Crime type severity ──
        # .map() replaces each primary_type with its weight; unmatched
        # types fall through to NaN, which .fillna() catches with the default.
        self.snapped["crime_type_weight"] = (
            crime_data["primary_type"]
            .map(self.CRIME_TYPE_WEIGHTS)
            .fillna(self._DEFAULT_CRIME_WEIGHT)
            .values
        )

        # ── 2. Temporal multiplier (based on hour of day) ──
        # np.select evaluates conditions top-to-bottom and picks the first
        # match, like a vectorized if/elif chain.
        hours = crime_data["hour"].values
        temporal_conditions = [
            (hours >= 22) | (hours <= 4),   # Late night / early morning
            (hours >= 18) & (hours <= 21),  # Evening
        ]
        temporal_choices = [1.8, 1.3]
        self.snapped["temporal_multiplier"] = np.select(
            temporal_conditions, temporal_choices, default=1.0
        )

        # ── 3. Day-of-week multiplier ──
        days = crime_data["day_of_week"].values
        day_conditions = [
            np.isin(days, [4, 5]),  # Friday, Saturday
            days == 6,              # Sunday
        ]
        day_choices = [1.2, 1.1]
        self.snapped["day_multiplier"] = np.select(
            day_conditions, day_choices, default=1.0
        )

        # ── Final composite weight ──
        self.snapped["final_weight"] = (
            self.snapped["crime_type_weight"]
            * self.snapped["temporal_multiplier"]
            * self.snapped["day_multiplier"]
        )

        logger.info(
            "Weights computed. Mean=%.2f, Min=%.2f, Max=%.2f",
            self.snapped["final_weight"].mean(),
            self.snapped["final_weight"].min(),
            self.snapped["final_weight"].max(),
        )
