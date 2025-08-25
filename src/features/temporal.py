import pandas as pd


class Temporal:
    MORNING_RUSH_START = 7
    MORNING_RUSH_END = 9
    EVENING_RUSH_START = 17
    EVENING_RUSH_END = 19
    WEEKEND_DAY = 5

    def create(self, df: pd.DataFrame, rush_hours: bool = True) -> pd.DataFrame:
        time_df = df.copy()
        time_df["datetime"] = pd.to_datetime(time_df["datetime"])
        return self.add_features(rush_hours, time_df)

    def add_features(self, rush_hours: bool, time_df: pd.DataFrame) -> pd.DataFrame:
        time_df["hour"] = self.get_hour(time_df)
        time_df["month"] = self.get_month(time_df)
        time_df["day_of_week"] = self.get_day_of_week(time_df)
        time_df["is_weekend"] = self.is_weekend(time_df)
        if rush_hours:
            time_df = self.add_rush_hour_features(time_df)

        return time_df

    def add_rush_hour_features(self, time_df) -> pd.DataFrame:
        time_df["is_morning_rush"] = self.is_mourning_rush(time_df)
        time_df["is_evening_rush"] = self.is_evening_rush(time_df)
        time_df["is_rush_hour"] = self.is_rush_hour(time_df)
        return time_df

    def is_mourning_rush(self, time_df):
        return (
            (time_df["hour"] >= self.MORNING_RUSH_START)
            & (time_df["hour"] <= self.MORNING_RUSH_END)
        ).astype(int)

    def is_evening_rush(self, time_df):
        return (
            (time_df["hour"] >= self.EVENING_RUSH_START)
            & (time_df["hour"] <= self.EVENING_RUSH_END)
        ).astype(int)

    def is_rush_hour(self, time_df):
        return (time_df["is_morning_rush"] | time_df["is_evening_rush"]).astype(int)

    def get_hour(self, time_df: pd.DataFrame) -> pd.Series:
        return time_df["datetime"].dt.hour

    def get_month(self, time_df: pd.DataFrame) -> pd.Series:
        return time_df["datetime"].dt.month

    def get_day_of_week(self, time_df: pd.DataFrame) -> pd.Series:
        return time_df["datetime"].dt.dayofweek

    def is_weekend(self, time_df: pd.DataFrame) -> pd.Series:
        return (time_df["day_of_week"] >= self.WEEKEND_DAY).astype(int)
