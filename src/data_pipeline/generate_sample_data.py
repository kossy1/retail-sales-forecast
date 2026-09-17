"""Generate synthetic retail sales data for testing."""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config.settings import config
from config.logging_config import logger


def generate_sample_sales(days: int = 730, n_stores: int = 5, n_skus: int = 20) -> pd.DataFrame:
    """Generate realistic retail sales data."""
    logger.info(f"Generating {days} days of data for {n_stores} stores × {n_skus} SKUs")

    np.random.seed(42)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]

    records = []
    for store_id in range(1, n_stores + 1):
        for sku_id in range(1, n_skus + 1):
            # Each SKU has a base demand
            base_demand = np.random.uniform(5, 100)
            base_price = np.random.uniform(5, 50)

            for date in dates:
                # Seasonality: weekly + yearly
                weekly = 1.0 + 0.3 * np.sin(2 * np.pi * date.weekday() / 7)
                yearly = 1.0 + 0.2 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365)

                # Trend (slight growth)
                trend = 1.0 + 0.0002 * (date - start_date).days

                # Random promotion (5% chance)
                is_promo = np.random.random() < 0.05
                promo_lift = 1.5 if is_promo else 1.0

                # Random noise
                noise = np.random.normal(1.0, 0.15)

                # Final sales
                sales = max(0, int(base_demand * weekly * yearly * trend * promo_lift * noise))

                # Price with small variation
                price = base_price * (0.9 if is_promo else 1.0) * np.random.uniform(0.95, 1.05)

                # Inventory (roughly 3x weekly demand)
                inventory = max(0, int(base_demand * 21 * np.random.uniform(0.5, 1.5)))

                records.append({
                    "date": date,
                    "store_id": store_id,
                    "sku_id": sku_id,
                    "sales_units": sales,
                    "price": round(price, 2),
                    "promotion_flag": int(is_promo),
                    "inventory_level": inventory,
                })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df):,} records")

    # Save to raw data
    output_path = config.data.raw_data_path / "sales_sample.parquet"
    df.to_parquet(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    generate_sample_sales()