import pandas as pd
from sklearn.cluster import KMeans


class Clustering:

    def __init__(self, n_clusters: int = 3, random_state: int = 95):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.start_kmeans = self.end_kmeans = None

    def create_columns(
        self, df: pd.DataFrame, kmeans_models: tuple[KMeans] = None
    ) -> pd.DataFrame:
        df_clustered = df.copy()
        self.define_models(df_clustered, kmeans_models)

        df_clustered = self.predict(df_clustered, "start")
        df_clustered = self.predict(df_clustered, "end")

        return df_clustered

    def define_models(self, df_clustered, kmeans_models):
        self.start_kmeans = (
            self.fit(df_clustered, "start")
            if kmeans_models is None
            else kmeans_models[0]
        )
        self.end_kmeans = (
            self.fit(df_clustered, "end") if kmeans_models is None else kmeans_models[1]
        )

    def predict(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        model = self.start_kmeans if column == "start" else self.end_kmeans
        coordinates = self.get_cordinates(column, df)
        df[f"{column}_cluster"] = model.predict(coordinates)
        return df

    def fit(self, df: pd.DataFrame, column: str) -> KMeans:
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        coordinates = self.get_cordinates(column, df)
        kmeans.fit(coordinates)
        return kmeans

    def get_cordinates(self, column, df):
        coordinates = df[[f"{column}_lat", f"{column}_lng"]].values
        return coordinates

    def get_models(self) -> tuple:
        return self.start_kmeans, self.end_kmeans
