"""Load raw data and clean it."""

import pandas as pd
import numpy as np
from config.settings import config
from config.logging_config import logger


def load_and_clean_data() -> pd.DataFrame:
    """Load raw data, clean it, and save to processed folder."""
    logger.info("Loading raw data...")

    raw_path = config.data.raw_data_path / "sales_sample.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No data found at {raw_path}. "
            "Run: python -m src.data_pipeline.generate_sample_data"
        )

    df = pd.read_parquet(raw_path)
    logger.info(f"Loaded {len(df):,} rows")

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    # Sort
    df = df.sort_values(["store_id", "sku_id", "date"]).reset_index(drop=True)

    # Handle stockouts
    df["is_stockout"] = (df["inventory_level"] == 0) & (df["sales_units"] == 0)

    # Cap outliers (99.5th percentile per SKU)
    upper = df.groupby("sku_id")["sales_units"].transform(lambda x: x.quantile(0.995))
    df["sales_units"] = np.minimum(df["sales_units"], upper)

    # Add time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["year"] = df["date"].dt.year
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Save processed
    output_path = config.data.processed_data_path / "sales_clean.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved cleaned data to {output_path}")

    return df


if __name__ == "__main__":
    load_and_clean_data()