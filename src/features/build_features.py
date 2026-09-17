"""Build features for modeling."""

import pandas as pd
import numpy as np
from typing import Tuple, List
from config.settings import config
from config.logging_config import logger


class FeatureBuilder:
    def __init__(self):
        self.group_cols = config.features.group_by_columns
        self.target = "sales_units"

    def build(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create all features."""
        logger.info("Building features...")
        df = df.copy().sort_values(self.group_cols + ["date"]).reset_index(drop=True)

        # Lag features
        for lag in config.features.lag_days:
            df[f"sales_lag_{lag}"] = df.groupby(self.group_cols)[self.target].shift(lag)

        # Rolling stats
        for window in config.features.rolling_windows:
            df[f"sales_roll_mean_{window}"] = df.groupby(self.group_cols)[self.target].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            df[f"sales_roll_std_{window}"] = df.groupby(self.group_cols)[self.target].transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )

        # Price lag
        df["price_lag_1"] = df.groupby(self.group_cols)["price"].shift(1)

        # Cyclical encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

               # Days since last promo
        # Step 1: Mark promo dates (or NaT if not a promo day)
        df["_promo_date"] = df["date"].where(df["promotion_flag"] == 1)

        # Step 2: Forward-fill within each store-SKU group
        df["last_promo_date"] = (
            df.groupby(self.group_cols)["_promo_date"]
            .transform(lambda s: s.ffill())
        )

        # Step 3: Compute days since
        df["days_since_promo"] = (
            (df["date"] - df["last_promo_date"]).dt.days.fillna(365).clip(0, 365)
        )

        # Drop the temporary column
        df = df.drop(columns=["_promo_date"])

        # Drop rows with missing lags
        df = df.dropna(subset=[f"sales_lag_{max(config.features.lag_days)}"]).reset_index(drop=True)

        # Feature columns
        exclude = ["date", "last_promo_date", self.target, "is_stockout"]
        feature_cols = [c for c in df.columns if c not in exclude]

        logger.info(f"Built {len(feature_cols)} features, {len(df):,} rows")
        return df, feature_cols


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Convenience function."""
    return FeatureBuilder().build(df)