from uuid import uuid4
import warnings
import pandas as pd
import haversine as hs

warnings.filterwarnings("ignore")


class TrajectoryDatabase:
    def __init__(self):
        self.trajectories = pd.DataFrame()
        self.segments = pd.DataFrame()
        self.trip_stats = pd.DataFrame()

    def store_trajectory(self, vehicle_id, points, timestamps, metadata=None):
        trajectory_id = uuid4()
        trip_data = {
            "trajectory_id": trajectory_id,
            "vehicle_id": vehicle_id,
            "start_time": timestamps[0],
            "end_time": timestamps[-1],
            "duration": (timestamps[-1] - timestamps[0]).total_seconds(),
            "start_lat": points[0][0],
            "start_lng": points[0][1],
            "end_lat": points[-1][0],
            "end_lng": points[-1][1],
            "num_points": len(points),
        }

        if metadata:
            for key, value in metadata.items():
                trip_data[key] = value

        self.trajectories = pd.concat(
            [self.trajectories, pd.DataFrame([trip_data])], ignore_index=True
        )

        for i in range(len(points) - 1):
            segment = {
                "trajectory_id": trajectory_id,
                "segment_id": i,
                "start_lat": points[i][0],
                "start_lng": points[i][1],
                "end_lat": points[i + 1][0],
                "end_lng": points[i + 1][1],
                "start_time": timestamps[i],
                "end_time": timestamps[i + 1],
                "duration": (timestamps[i + 1] - timestamps[i]).total_seconds(),
                "distance": hs.haversine(points[i], points[i + 1]),
            }

            if segment["duration"] > 0:
                segment["speed"] = (segment["distance"] / segment["duration"]) * 3600
            else:
                segment["speed"] = 0

            self.segments = pd.concat(
                [self.segments, pd.DataFrame([segment])], ignore_index=True
            )

        return trajectory_id

    def query_similar_trips(
        self, start_lat, start_lng, end_lat, end_lng, time_of_day=None, limit=5
    ):
        if len(self.trajectories) == 0:
            return pd.DataFrame()

        self.trajectories["start_distance"] = self.trajectories.apply(
            lambda row: hs.haversine(
                (row["start_lat"], row["start_lng"]), (start_lat, start_lng)
            ),
            axis=1,
        )

        self.trajectories["end_distance"] = self.trajectories.apply(
            lambda row: hs.haversine(
                (row["end_lat"], row["end_lng"]), (end_lat, end_lng)
            ),
            axis=1,
        )

        similar_trips = self.trajectories[
            (self.trajectories["start_distance"] < 1)
            & (self.trajectories["end_distance"] < 1)
        ].copy()

        if time_of_day and len(similar_trips) > 0:
            similar_trips["hour_diff"] = similar_trips["start_time"].apply(
                lambda x: min(
                    abs((x.hour - time_of_day.hour) % 24),
                    abs((time_of_day.hour - x.hour) % 24),
                )
            )
            similar_trips = similar_trips[similar_trips["hour_diff"] <= 2]

        if len(similar_trips) > 0:
            similar_trips["total_distance"] = (
                similar_trips["start_distance"] + similar_trips["end_distance"]
            )
            return similar_trips.sort_values("total_distance").head(limit)
        else:
            return pd.DataFrame()

    def get_statistics(self):
        stats = {
            "total_trajectories": len(self.trajectories),
            "total_segments": len(self.segments),
            "avg_duration": (
                self.trajectories["duration"].mean()
                if len(self.trajectories) > 0
                else 0
            ),
            "avg_speed": self.segments["speed"].mean() if len(self.segments) > 0 else 0,
            "total_distance": (
                self.segments["distance"].sum() if len(self.segments) > 0 else 0
            ),
        }
        return stats
