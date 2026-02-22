"""
SafeStride — Step 4, Part 2: Safe Route Router
===============================================
Wraps the scored NetworkX graph produced by CrimeScorer.run() and
exposes a simple safe-routing interface:

  1. set_edge_weights()  — blend crime risk + distance into one cost
  2. find_route()        — shortest path on the blended cost

Why a blended weight?
  Pure crime-minimisation ignores distance — it might route you 3 km
  out of the way to avoid one slightly risky block. Pure distance-
  minimisation ignores safety entirely. The alpha/beta blend lets the
  caller dial the trade-off: alpha=1, beta=0 → purely avoid crime;
  alpha=0, beta=1 → pure shortest path; default 0.7/0.3 → safety-first
  but still practical.
"""

import logging

import networkx as nx
import osmnx as ox

logger = logging.getLogger("safestride.router")


class SafeStrideRouter:
    """
    Computes pedestrian-safe routes on a crime-scored NetworkX graph.

    Parameters fed in from CrimeScorer.run() — the graph already has
    crime_risk_score and length on every edge.
    """

    def __init__(self, graph) -> None:
        """
        Accept the scored graph from CrimeScorer.run().

        Parameters
        ----------
        graph : networkx.MultiDiGraph
            Chicago walk network with crime_risk_score ∈ [0, 1]
            and length (metres) on every edge.
        """
        self.graph = graph
        logger.info(
            "SafeStrideRouter initialised with %d nodes and %d edges.",
            len(self.graph.nodes),
            len(self.graph.edges),
        )

    # ──────────────────────────────────────────
    # EDGE WEIGHT COMPUTATION
    # ──────────────────────────────────────────
    def set_edge_weights(self, alpha: float = 0.7, beta: float = 0.3) -> None:
        """
        Compute a blended route_weight for every edge and write it in-place.

        Formula
        -------
            route_weight = alpha × crime_risk_score
                         + beta  × (length / max_length)

        Components
        ----------
        crime_risk_score : float ∈ [0, 1]
            Normalised danger score from CrimeScorer (0 = safe, 1 = dangerous).
            Already attached to every edge by CrimeScorer.attach_to_graph().

        length / max_length : float ∈ [0, 1]
            OSMnx stores edge length in metres. We divide by the global
            maximum so it's unit-agnostic and comparable to crime_risk_score.
            A longer segment costs more — biases the router toward shorter hops
            within similarly-safe areas.

        alpha / beta
        ------------
        Must not be zero simultaneously. They do NOT need to sum to 1 —
        treating them as independent importance multipliers gives more
        intuitive tuning (e.g. alpha=1, beta=0 = crime-only, no length bias).

        Parameters
        ----------
        alpha : float
            Weight on crime risk. Default 0.7 (safety-first).
        beta : float
            Weight on normalised length. Default 0.3 (distance awareness).
        """
        if alpha == 0 and beta == 0:
            raise ValueError("alpha and beta cannot both be zero.")

        logger.info(
            "Computing route_weight for %d edges (alpha=%.2f, beta=%.2f)...",
            len(self.graph.edges),
            alpha,
            beta,
        )

        # ── Find max edge length for normalization ──────────────────────────
        # Collect all lengths across all edges (MultiDiGraph: graph[u][v][k]).
        # 'length' is added by OSMnx in metres; default to 1.0 if missing.
        all_lengths = [
            data.get("length", 1.0)
            for _, _, data in self.graph.edges(data=True)
        ]
        max_length = max(all_lengths) if all_lengths else 1.0

        if max_length == 0:
            max_length = 1.0  # Edge case: avoid divide-by-zero

        logger.info("Max edge length across graph: %.2f m", max_length)

        # ── Write route_weight on every edge ───────────────────────────────
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            crime_score  = data.get("crime_risk_score", 0.0)
            norm_length  = data.get("length", 1.0) / max_length

            self.graph[u][v][key]["route_weight"] = (
                alpha * crime_score + beta * norm_length
            )

        logger.info("route_weight attached to all edges.")

    # ──────────────────────────────────────────
    # ROUTE FINDING
    # ──────────────────────────────────────────
    def find_route(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> list[int]:
        """
        Find the safest route between two GPS coordinates.

        Steps
        -----
        1. Snap origin and destination to the nearest graph nodes using
           osmnx.nearest_nodes — this handles the gap between a raw GPS
           point (e.g. middle of a block) and the graph's intersection nodes.

        2. Run networkx.shortest_path with weight="route_weight" — this is
           Dijkstra's algorithm, guaranteed to find the global minimum-cost
           path (not just a local heuristic like A* without a good heuristic).

        Parameters
        ----------
        origin_lat, origin_lon : float
            Start point in WGS-84 decimal degrees.
        dest_lat, dest_lon : float
            End point in WGS-84 decimal degrees.

        Returns
        -------
        list[int]
            Ordered list of OSMnx node IDs forming the route.
            Pass to osmnx.plot_route_folium() or the routing API.

        Raises
        ------
        RuntimeError
            If route_weight hasn't been computed yet (set_edge_weights not called).
        networkx.NetworkXNoPath
            If no path exists between origin and destination nodes.
        networkx.NodeNotFound
            If snapped nodes are not in the graph (shouldn't happen with
            osmnx.nearest_nodes, but guarded for robustness).
        """
        # Guard: ensure weights have been set before routing.
        sample_edge = next(iter(self.graph.edges(data=True)), None)
        if sample_edge and "route_weight" not in sample_edge[2]:
            raise RuntimeError(
                "route_weight not found on edges. "
                "Call set_edge_weights() before find_route()."
            )

        logger.info(
            "Finding route: (%.6f, %.6f) → (%.6f, %.6f)",
            origin_lat, origin_lon, dest_lat, dest_lon,
        )

        # ── Snap GPS coords to nearest graph nodes ──────────────────────────
        # ox.nearest_nodes expects (X=longitude, Y=latitude) — note the order.
        origin_node = ox.nearest_nodes(self.graph, X=origin_lon, Y=origin_lat)
        dest_node   = ox.nearest_nodes(self.graph, X=dest_lon,   Y=dest_lat)

        logger.info(
            "Snapped to nodes: origin=%d, destination=%d",
            origin_node, dest_node,
        )

        # ── Dijkstra shortest path on blended edge weights ──────────────────
        route = nx.shortest_path(
            self.graph,
            source=origin_node,
            target=dest_node,
            weight="route_weight",
        )

        logger.info(
            "Route found: %d nodes, %.0f m total length (approx).",
            len(route),
            sum(
                self.graph[route[i]][route[i + 1]][0].get("length", 0)
                for i in range(len(route) - 1)
            ),
        )

        return route

    # ──────────────────────────────────────────
    # PARETO ROUTE GENERATION
    # ──────────────────────────────────────────
    def find_pareto_routes(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
    ) -> list[dict]:
        """
        Generate three Pareto-optimal route alternatives by varying the
        safety ↔ speed trade-off parameter.

        Each alternative re-runs set_edge_weights() with a different alpha/beta
        ratio and then calls find_route(), so the graph edges are re-weighted
        for each scenario before the Dijkstra search.

        Presets
        -------
        | Label              | alpha | beta | Behaviour                        |
        |--------------------|-------|------|----------------------------------|
        | safety_maximized   |  0.9  |  0.1 | Strongly avoids high-crime edges |
        | balanced           |  0.5  |  0.5 | Equal safety and speed bias      |
        | speed_maximized    |  0.1  |  0.9 | Prefers shortest total distance  |

        Parameters
        ----------
        origin_lat, origin_lon : float
            Start point in WGS-84 decimal degrees.
        dest_lat, dest_lon : float
            End point in WGS-84 decimal degrees.

        Returns
        -------
        list[dict]
            Three dicts, each with keys:
                - ``label``  : human-readable preset name (str)
                - ``alpha``  : crime-risk weight used (float)
                - ``beta``   : distance weight used (float)
                - ``route``  : ordered list of OSMnx node IDs (list[int])
        """
        presets = [
            {"label": "safety_maximized", "alpha": 0.9, "beta": 0.1},
            {"label": "balanced",         "alpha": 0.5, "beta": 0.5},
            {"label": "speed_maximized",  "alpha": 0.1, "beta": 0.9},
        ]

        results = []
        for preset in presets:
            logger.info(
                "Computing Pareto route '%s' (alpha=%.1f, beta=%.1f)...",
                preset["label"],
                preset["alpha"],
                preset["beta"],
            )
            # Re-weight edges for this scenario before routing.
            self.set_edge_weights(alpha=preset["alpha"], beta=preset["beta"])
            route = self.find_route(origin_lat, origin_lon, dest_lat, dest_lon)

            results.append(
                {
                    "label": preset["label"],
                    "alpha": preset["alpha"],
                    "beta":  preset["beta"],
                    "route": route,
                }
            )
            logger.info(
                "Pareto route '%s': %d nodes.",
                preset["label"],
                len(route),
            )

        logger.info("All 3 Pareto routes computed successfully.")
        return results
