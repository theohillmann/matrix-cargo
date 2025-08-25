import pandas as pd


class Geographical:

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        geo_df = df.copy()
        geo_df["lat_diff"] = self.get_geografic_delta(geo_df, "lat")
        geo_df["lng_diff"] = self.get_geografic_delta(geo_df, "lng")
        return geo_df

    def get_geografic_delta(
        self, geo_df: pd.DataFrame, coordinate_type: str
    ) -> pd.Series:
        return geo_df[f"end_{coordinate_type}"] - geo_df[f"start_{coordinate_type}"]
