import os
import pandas as pd
from joblib import load
from src.features import (
    Clustering,
    DistanceCalculator,
    Geographical,
    Interactions,
    Temporal,
)


class FeaturePipeline:
    def __init__(self):
        self.clustering = Clustering()
        self.distance_calculator = DistanceCalculator()
        self.geographical = Geographical()
        self.interactions = Interactions()
        self.temporal = Temporal()

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        df_features = df.copy()
        df_features.dropna(inplace=True)
        # df_features.drop("row_id", axis=1, inplace=True)
        df_features = self.clustering.create_columns(
            df_features, self.load_kmeas_models()
        )
        df_features = self.distance_calculator.calculate(df_features)
        df_features = self.geographical.create(df_features)
        df_features = self.interactions.create(df_features)
        df_features = self.temporal.create(df_features)
        return df_features

    def load_kmeas_models(self) -> tuple:
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        start_kmeans_path = os.path.join(
            root_path, "models/production/start_cluster_model.pkl"
        )
        end_kmeans_path = os.path.join(
            root_path, "models/production/end_cluster_model.pkl"
        )
        return load(start_kmeans_path), load(end_kmeans_path)
