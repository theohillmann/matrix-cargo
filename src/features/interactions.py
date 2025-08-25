import pandas as pd


class Interactions:

    def create(self, df: pd.DataFrame) -> pd.DataFrame:
        interaction_df = df.copy()
        interaction_df = self.add_same_cluster_features(interaction_df)

        return interaction_df

    def add_same_cluster_features(self, df: pd.DataFrame):
        df["is_same_cluster"] = (df["start_cluster"] == df["end_cluster"]).astype(int)
        df["is_inter_cluster"] = (df["start_cluster"] != df["end_cluster"]).astype(int)
        return df
