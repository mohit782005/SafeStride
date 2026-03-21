import os
import requests
import logging
from dotenv import load_dotenv

class WeatherScorer:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENWEATHER_API_KEY is missing")
            
        self.logger = logging.getLogger("safestride.weather_scorer")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def fetch_weather(self, lat=41.8781, lon=-87.6298):
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}&units=metric"
        
        try:
            response = requests.get(url)
            if not response.ok:
                raise RuntimeError(f"API call failed with status: {response.status_code}, response: {response.text}")
                
            data = response.json()
            
            condition = data.get("weather", [{}])[0].get("main", "Unknown")
            self.logger.info(f"Fetched weather condition: {condition}")
            
            dt = data.get("dt", 0)
            sys_data = data.get("sys", {})
            sunrise = sys_data.get("sunrise", 0)
            sunset = sys_data.get("sunset", 0)
            
            is_night = False
            if sunrise and sunset and dt:
                is_night = dt < sunrise or dt > sunset
                
            return {
                "condition": condition,
                "temp_c": float(data.get("main", {}).get("temp", 0.0)),
                "visibility_m": int(data.get("visibility", 10000)),
                "wind_speed_ms": float(data.get("wind", {}).get("speed", 0.0)),
                "is_night": is_night
            }
            
        except Exception as e:
            self.logger.warning(f"Weather API unavailable: {e}")
            self.logger.info("Falling back to default 'Clear' daytime weather.")
            return {
                "condition": "Clear",
                "temp_c": 20.0,
                "visibility_m": 10000,
                "wind_speed_ms": 0.0,
                "is_night": False
            }

    def compute_weather_multiplier(self, weather: dict) -> float:
        condition = weather.get("condition", "Unknown")
        visibility = weather.get("visibility_m", 10000)
        is_night = weather.get("is_night", False)

        # Condition multiplier
        if condition == "Thunderstorm":
            cond_mult = 1.8
        elif condition == "Snow":
            cond_mult = 1.6
        elif condition in ["Fog", "Mist", "Haze"]:
            cond_mult = 1.5
        elif condition in ["Rain", "Drizzle"]:
            cond_mult = 1.4
        elif condition in ["Dust", "Sand"]:
            cond_mult = 1.3
        elif condition == "Clear":
            cond_mult = 1.0
        else:
            cond_mult = 1.1

        # Visibility multiplier
        if visibility < 1000:
            vis_mult = 1.4
        elif visibility <= 4000:
            vis_mult = 1.2
        else:
            vis_mult = 1.0

        # Night multiplier
        night_mult = 1.3 if is_night else 1.0

        # Final multiplier capped at 3.0
        multiplier = cond_mult * vis_mult * night_mult
        final_multiplier = min(multiplier, 3.0)

        self.logger.info(f"Computed weather multiplier: {final_multiplier:.2f} (cond={cond_mult}, vis={vis_mult}, night={night_mult})")
        return final_multiplier
