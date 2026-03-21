import requests

BASE_URL = "http://127.0.0.1:8000"

def test_api():
    print("Testing /health ...")
    try:
        health_resp = requests.get(f"{BASE_URL}/health")
        print(f"Status {health_resp.status_code}: {health_resp.json()}")
    except requests.exceptions.ConnectionError:
        print("API is not running. Start it with: uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
        return
        
    print("\nTesting /routes ...")
    payload = {
        "origin_lat": 41.8827,
        "origin_lon": -87.6233,
        "dest_lat": 41.8750,
        "dest_lon": -87.6350
    }
    
    routes_resp = requests.post(f"{BASE_URL}/routes", json=payload)
    if routes_resp.status_code == 200:
        data = routes_resp.json()
        routes = data.get("routes", [])
        print(f"Success! Received {len(routes)} routes.\n")
        
        for idx, route in enumerate(routes):
            print(f"[{idx}] {route['label']}")
            print(f"    Distance: {route['total_distance_m']} m")
            print(f"    Time:     {route['estimated_time_min']} min")
            geom = route.get("geometry", [])
            print(f"    Geometry: {geom[:3]} ... (truncated)")
            print()
    else:
        print(f"Failed /routes: {routes_resp.status_code} - {routes_resp.text}")

if __name__ == "__main__":
    test_api()
