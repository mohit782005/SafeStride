import subprocess
import time
import requests
import sys

print("Starting uvicorn server...")
server = subprocess.Popen([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

max_retries = 90
ready = False
for i in range(max_retries):
    try:
        r = requests.get("http://127.0.0.1:8000/health")
        if r.status_code == 200:
            ready = True
            break
    except Exception:
        pass
    if i % 5 == 0:
        print(f"Waiting for server setup... ({i*2}s elapsed)")
    time.sleep(2)

if not ready:
    print("Server failed to start in time.")
    server.terminate()
    print(server.stdout.read().decode('utf-8'))
    sys.exit(1)

print("Server is ready, sending POST /routes payload...")
payload = {
    "origin_lat": 41.8827,
    "origin_lon": -87.6233,
    "dest_lat": 41.8750,
    "dest_lon": -87.6350
}
result = requests.post("http://127.0.0.1:8000/routes", json=payload)

print(f"Response Status: {result.status_code}")
if result.status_code == 200:
    data = result.json()
    routes = data.get("routes", [])
    print(f"Received {len(routes)} routes.")
    for i, route in enumerate(routes):
        geom_points = len(route.get('geometry', []))
        print(f"  Route {i} ({route['label']}):")
        print(f"    - distance: {route['total_distance_m']}m")
        print(f"    - avg_safety_score: {route['avg_safety_score']}")
        print(f"    - geometry length: {geom_points} points")
        if geom_points > 0:
            print(f"    - First point: {route['geometry'][0]}")
            print(f"    - Last point: {route['geometry'][-1]}")
else:
    print(f"Error payload: {result.text}")

server.terminate()
