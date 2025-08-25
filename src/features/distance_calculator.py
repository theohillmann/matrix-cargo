import numpy as np
import pandas as pd


class DistanceCalculator:
    EARTH_RADIUS_KM = 6371.0

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        df_dist = df.copy()
        df_dist["distance_km"] = self.haversine_distance(df)
        return df_dist

    def haversine_distance(self, df: pd.DataFrame):
        start_coordinates = self.get_coordinates(df, "start")
        end_coordinates = self.get_coordinates(df, "end")
        dlat, dlon = self.get_delta_coordinates(start_coordinates, end_coordinates)
        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(start_coordinates[0])
            * np.cos(end_coordinates[0])
            * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        return c * self.EARTH_RADIUS_KM

    def get_delta_coordinates(
        self, start_coordinates: tuple, end_coordinates: tuple
    ) -> tuple:
        dlat = end_coordinates[0] - start_coordinates[0]
        dlon = end_coordinates[1] - start_coordinates[1]
        return dlat, dlon

    def get_coordinates(self, df: pd.DataFrame, column: str) -> tuple:
        lat = np.radians(df[f"{column}_lat"])
        lon = np.radians(df[f"{column}_lng"])
        return lat, lon
