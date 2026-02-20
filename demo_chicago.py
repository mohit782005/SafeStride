"""
SafeStride — Chicago Road Network Demo
Pulls the drivable road network of Chicago using OSMnx,
plots it, and displays it to confirm the environment works.
"""
import osmnx as ox
import matplotlib.pyplot as plt

print("📍 Downloading Chicago road network (this may take a minute)...")
G = ox.graph_from_place("Chicago, Illinois, USA", network_type="drive")

print(f"✅ Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

fig, ax = ox.plot_graph(
    G, figsize=(12, 12), node_size=0,
    edge_color="white", edge_linewidth=0.3,
    bgcolor="black", show=False, close=False
)
ax.set_title("Chicago Road Network — SafeStride", color="white", fontsize=16)
plt.tight_layout()
print("🗺️  Displaying plot... Close the window to exit.")
plt.show()
