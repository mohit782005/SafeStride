import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from src.safety.crime_scorer import CrimeScorer
from src.routing.router import SafeStrideRouter

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safestride.api")

app = FastAPI(title="SafeStride API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("Starting up SafeStride API...")
    
    # Load and score graph
    logger.info("Running CrimeScorer.run() to load the scored graph...")
    scorer = CrimeScorer()
    graph = scorer.run()
    app.state.graph = graph
    
    # Initialize the router and weights
    logger.info("Instantiating SafeStrideRouter and setting edge weights...")
    router = SafeStrideRouter(graph)
    router.set_edge_weights()
    app.state.router = router
    
    logger.info("SafeStride API ready")

@app.get("/health")
def health_check():
    return {"status": "ok"}

class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float

@app.post("/routes")
def get_routes(request: RouteRequest):
    routes_data = app.state.router.find_pareto_routes(
        origin_lat=request.origin_lat,
        origin_lon=request.origin_lon,
        dest_lat=request.dest_lat,
        dest_lon=request.dest_lon
    )
    
    formatted_routes = []
    for r in routes_data:
        # Calculate edge count from the node list
        edge_count = max(0, len(r["route"]) - 1)
        
        # Get route geometry
        geometry = app.state.router.get_route_geometry(r["route"])
        
        formatted_routes.append({
            "label": r["label"],
            "total_distance_m": r["total_distance_m"],
            "estimated_time_min": r["estimated_time_min"],
            "avg_safety_score": r["avg_safety_score"],
            "max_danger_score": r["max_danger_score"],
            "edge_count": edge_count,
            "geometry": geometry
        })
        
    return {"routes": formatted_routes}
