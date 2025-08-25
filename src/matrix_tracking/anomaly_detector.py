import haversine as hs
from src.predict.duration_preditcor import DurationPredictor
import warnings

warnings.filterwarnings("ignore")


class AnomalyDetector:
    def __init__(self, threshold=0.8):
        self.duration_predictor = DurationPredictor()
        self.threshold = threshold

    def detect_time_anomalies(self, actual_duration, trip_data):
        start_lat = trip_data["start_lat"]
        start_lng = trip_data["start_lng"]
        end_lat = trip_data["end_lat"]
        end_lng = trip_data["end_lng"]
        datetime_str = trip_data["datetime"]

        predicted_duration = self.duration_predictor.predict(
            start_lng, start_lat, end_lng, end_lat, datetime_str
        )[0]

        deviation = float(
            (actual_duration - predicted_duration) / predicted_duration
            if predicted_duration > 0
            else 0
        )

        is_anomaly = abs(deviation) > self.threshold
        anomaly_type = None
        if is_anomaly:
            if deviation > 0:
                anomaly_type = "delay"
            else:
                anomaly_type = "early"

        return {
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "actual_duration": actual_duration,
            "predicted_duration": float(predicted_duration),
            "deviation": deviation,
        }

    def detect_route_anomalies(self, current_position, planned_route, max_distance=0.5):
        min_distance = float("inf")
        nearest_point = None

        for point in planned_route["waypoints"]:
            distance = hs.haversine(current_position, point)
            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        is_route_anomaly = min_distance > max_distance

        return {
            "is_anomaly": is_route_anomaly,
            "distance_from_route": min_distance,
            "nearest_point": nearest_point,
        }
