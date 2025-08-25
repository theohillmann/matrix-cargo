import pandas as pd
import haversine as hs
from .anomaly_detector import AnomalyDetector
from .trajectory_database import TrajectoryDatabase
from .routing_engine import routing_engine_calculate_route


class MatrixTrackingSystem:
    def __init__(self):
        self.trajectory_db = TrajectoryDatabase()
        self.anomaly_detector = AnomalyDetector()
        self.active_vehicles = {}
        self.alerts = []

    def plan_route(
        self, vehicle_id, start_lat, start_lng, end_lat, end_lng, departure_time
    ):
        route = routing_engine_calculate_route(
            start_lat, start_lng, end_lat, end_lng, departure_time
        )

        self.active_vehicles[vehicle_id] = {
            "vehicle_id": vehicle_id,
            "start_lat": start_lat,
            "start_lng": start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
            "departure_time": departure_time,
            "planned_route": route,
            "expected_duration": route["duration"],
            "expected_arrival": pd.to_datetime(departure_time)
            + pd.Timedelta(seconds=route["duration"]),
            "current_position": (start_lat, start_lng),
            "last_update": departure_time,
            "trajectory": [(start_lat, start_lng)],
            "timestamps": [pd.to_datetime(departure_time)],
            "status": "planned",
            "alerts": [],
        }

        return {
            "vehicle_id": vehicle_id,
            "planned_route": route,
            "expected_duration": route["duration"],
            "expected_arrival": pd.to_datetime(departure_time)
            + pd.Timedelta(seconds=route["duration"]),
        }

    def update_vehicle_position(self, vehicle_id, lat, lng, timestamp):
        if vehicle_id not in self.active_vehicles:
            return {"error": "Veículo não encontrado"}

        vehicle = self.active_vehicles[vehicle_id]
        timestamp = pd.to_datetime(timestamp)

        previous_position = vehicle["current_position"]
        vehicle["current_position"] = (lat, lng)
        vehicle["trajectory"].append((lat, lng))
        vehicle["timestamps"].append(timestamp)
        vehicle["last_update"] = timestamp
        vehicle["status"] = "active"

        distance_to_end = hs.haversine(
            (lat, lng), (vehicle["end_lat"], vehicle["end_lng"])
        )
        if distance_to_end < 0.1:
            vehicle["status"] = "completed"

            actual_duration = (
                timestamp - pd.to_datetime(vehicle["departure_time"])
            ).total_seconds()

            time_result = self.anomaly_detector.detect_time_anomalies(
                actual_duration,
                {
                    "start_lat": vehicle["start_lat"],
                    "start_lng": vehicle["start_lng"],
                    "end_lat": vehicle["end_lat"],
                    "end_lng": vehicle["end_lng"],
                    "datetime": vehicle["departure_time"],
                },
            )

            metadata = {
                "planned_duration": vehicle["expected_duration"],
                "actual_duration": actual_duration,
                "deviation": time_result["deviation"],
            }

            self.trajectory_db.store_trajectory(
                vehicle_id, vehicle["trajectory"], vehicle["timestamps"], metadata
            )

            if time_result["is_anomaly"]:
                alert = {
                    "vehicle_id": vehicle_id,
                    "timestamp": timestamp,
                    "type": "time_anomaly",
                    "details": f"Anomalia de tempo detectada: {time_result['anomaly_type']}. Desvio de {100*time_result['deviation']:.1f}%",
                }
                vehicle["alerts"].append(alert)
                self.alerts.append(alert)

            return {
                "status": "completed",
                "actual_duration": actual_duration,
                "expected_duration": vehicle["expected_duration"],
                "deviation": time_result["deviation"] * 100,
                "is_anomaly": time_result["is_anomaly"],
                "alerts": vehicle["alerts"],
            }

        route_result = self.anomaly_detector.detect_route_anomalies(
            (lat, lng), vehicle["planned_route"]
        )

        if route_result["is_anomaly"]:
            alert = {
                "vehicle_id": vehicle_id,
                "timestamp": timestamp,
                "type": "route_anomaly",
                "details": f"Desvio de rota detectado. Distância: {route_result['distance_from_route']:.2f} km da rota planejada.",
            }
            vehicle["alerts"].append(alert)
            self.alerts.append(alert)

        total_distance = hs.haversine(
            (vehicle["start_lat"], vehicle["start_lng"]),
            (vehicle["end_lat"], vehicle["end_lng"]),
        )
        progress = 1.0 - (distance_to_end / total_distance) if total_distance > 0 else 0
        progress = max(0, min(1, progress))

        time_elapsed = (
            timestamp - pd.to_datetime(vehicle["departure_time"])
        ).total_seconds()
        if progress > 0.05:
            estimated_total_time = time_elapsed / progress
            new_eta = pd.to_datetime(vehicle["departure_time"]) + pd.Timedelta(
                seconds=estimated_total_time
            )
            delay = estimated_total_time - vehicle["expected_duration"]
        else:
            new_eta = vehicle["expected_arrival"]
            delay = 0

        if delay > 300:
            alert = {
                "vehicle_id": vehicle_id,
                "timestamp": timestamp,
                "type": "delay_prediction",
                "details": f"Previsão de atraso: {delay/60:.1f} minutos. Nova ETA: {new_eta.strftime('%H:%M:%S')}",
            }
            vehicle["alerts"].append(alert)
            self.alerts.append(alert)

        return {
            "status": "active",
            "progress": progress * 100,
            "distance_to_end": distance_to_end,
            "current_position": (lat, lng),
            "time_elapsed": time_elapsed,
            "original_eta": vehicle["expected_arrival"],
            "new_eta": new_eta,
            "delay": delay,
            "route_deviation": route_result["is_anomaly"],
            "alerts": vehicle["alerts"],
        }

    def get_vehicle_status(self, vehicle_id=None):
        if vehicle_id:
            if vehicle_id in self.active_vehicles:
                return self.active_vehicles[vehicle_id]
            return {"error": "Veículo não encontrado"}
        else:
            return {
                vid: {"status": v["status"], "position": v["current_position"]}
                for vid, v in self.active_vehicles.items()
            }
