import warnings
import numpy as np
import pandas as pd
import haversine as hs
from datetime import datetime
from src.predict.duration_preditcor import DurationPredictor

warnings.filterwarnings("ignore")


def routing_engine_calculate_route(
    start_lat, start_lng, end_lat, end_lng, departure_time=None
):

    distance_km = hs.haversine((start_lat, start_lng), (end_lat, end_lng))

    duration_predictor = DurationPredictor()
    estimated_time = duration_predictor.predict(
        start_lng,
        start_lat,
        end_lng,
        end_lat,
        (
            departure_time
            if departure_time
            else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ),
    )[0]

    waypoints = simulate_route(distance_km, start_lat, start_lng, end_lat, end_lng)
    traffic_conditions = simulate_traffic_conditions(departure_time)

    return {
        "distance": distance_km,
        "duration": estimated_time,
        "waypoints": waypoints,
        "traffic_conditions": traffic_conditions,
    }


def simulate_traffic_conditions(departure_time):
    traffic_conditions = "unknown"
    if departure_time:
        hour = (
            departure_time.hour
            if isinstance(departure_time, datetime)
            else pd.to_datetime(departure_time).hour
        )
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            traffic_conditions = "heavy"
        else:
            traffic_conditions = "normal"
    return traffic_conditions


def simulate_route(distance_km, start_lat, start_lng, end_lat, end_lng):
    num_points = max(3, int(distance_km / 2))
    waypoints = []
    for i in range(num_points):
        frac = i / (num_points - 1)
        lat = start_lat + frac * (end_lat - start_lat)
        lng = start_lng + frac * (end_lng - start_lng)

        if 0 < i < num_points - 1:
            lat += np.random.normal(0, 0.005)
            lng += np.random.normal(0, 0.005)
        waypoints.append((lat, lng))
    return waypoints
