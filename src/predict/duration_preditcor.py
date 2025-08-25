import os.path
import pickle
import pandas as pd
from pathlib import Path
from src.pipeline import FeaturePipeline


class DurationPredictor:

    def __init__(self):
        self.feature_pipeline = FeaturePipeline()

    def predict(self, start_lng, start_lat, end_lng, end_lat, datetime):
        df = pd.DataFrame(
            {
                "start_lng": [start_lng],
                "start_lat": [start_lat],
                "end_lng": [end_lng],
                "end_lat": [end_lat],
                "datetime": [datetime],
            }
        )
        df = self.prepare_df(df)

        model = self.get_model()

        return model.predict(df)

    def get_model(self):
        base_dir = Path(__file__).resolve().parents[2]
        model_path = os.path.join(
            base_dir, "models", "production", "ronsomForestRefressor.pkl"
        )

        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def prepare_df(self, df):
        df = self.feature_pipeline.fit(df)
        df = df.sort_index(axis=1)
        df.drop("datetime", axis=1, inplace=True)
        return df
