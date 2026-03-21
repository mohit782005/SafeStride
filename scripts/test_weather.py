import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.safety.weather_scorer import WeatherScorer

def main():
    try:
        ws = WeatherScorer()
        weather = ws.fetch_weather()
        print("Weather Data:")
        print(weather)
        
        multiplier = ws.compute_weather_multiplier(weather)
        print(f"\nComputed Pedestrian Risk Multiplier: {multiplier:.2f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
