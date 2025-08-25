import pandas as pd
from .feature_pipeline import FeaturePipeline


class TrainPipeline:

    def __init__(self):
        self.feature_pipeline = FeaturePipeline()

    def fit(self, df):
        train_df = df.copy()
        print(f"Início: {len(df)} linhas")
        train_df.dropna(inplace=True)
        print(f"Após dropna: {len(train_df)} linhas")
        train_df = self.fix_coordinates(train_df)
        print(f"Após fix_coordinates: {len(train_df)} linhas")
        train_df = self.feature_pipeline.fit(train_df)
        print(f"Após feature_pipeline: {len(train_df)} linhas")
        train_df = self.remove_outliers(
            train_df, "distance_km", cluster_col="start_cluster"
        )
        print(f"Após remove_outliers start: {len(train_df)} linhas")
        train_df = self.remove_outliers(
            train_df, "distance_km", cluster_col="end_cluster"
        )
        print(f"Após remove_outliers end: {len(train_df)} linhas")
        train_df = train_df[train_df["duration"] > 0]
        print(f"Após duration > 0: {len(train_df)} linhas")
        train_df = train_df[train_df["distance_km"] > 0]
        print(f"Após distance_km > 0: {len(train_df)} linhas")

        train_df = train_df[
            train_df["distance_km"] / (self.impossiple_values(df)["duration"] / 60 / 60)
            < 100
        ]
        print(f"Após speed filter: {len(train_df)} linhas")

        return train_df

    def fix_coordinates(self, df):

        suspicious_start = df["start_lng"] > 0
        suspicious_end = df["end_lng"] > 0

        df.loc[suspicious_start, "start_lng"] = -df.loc[suspicious_start, "start_lng"]
        df.loc[suspicious_end, "end_lng"] = -df.loc[suspicious_end, "end_lng"]

        return df

    def remove_outliers(self, df, outlier_col, cluster_col="distance_km"):
        result = df.copy()
        mask_to_keep = pd.Series([True] * len(result), index=result.index)

        for cluster in df[cluster_col].unique():
            cluster_mask = df[cluster_col] == cluster
            cluster_data = df[cluster_mask][outlier_col]

            Q1, Q3 = cluster_data.quantile([0.25, 0.75])
            IQR = Q3 - Q1

            outlier_mask = cluster_mask & (
                (df[outlier_col] < Q1 - 1.5 * IQR) | (df[outlier_col] > Q3 + 1.5 * IQR)
            )
            mask_to_keep &= ~outlier_mask

        return result[mask_to_keep]

    def impossiple_values(self, df):
        impossible_values_df = df.copy()
        print(len(impossible_values_df))
        impossible_values_df = impossible_values_df[
            impossible_values_df["duration"] > 0
        ]
        impossible_values_df = impossible_values_df[
            impossible_values_df["duration"] < 10800
        ]
        return impossible_values_df
