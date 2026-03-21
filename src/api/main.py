import logging
from fastapi import FastAPI
from src.safety.crime_scorer import CrimeScorer
from src.routing.router import SafeStrideRouter

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("safestride.api")

app = FastAPI(title="SafeStride API", version="0.1.0")

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
