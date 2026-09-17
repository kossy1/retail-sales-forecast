"""Data quality checks for uploaded POS data."""

import pandas as pd
import numpy as np
from typing import Dict, Any


def analyze_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze uploaded DataFrame for quality issues."""
    report: Dict[str, Any] = {
        "total_rows": int(len(df)),
        "issues": {},
        "warnings": [],
    }

    if df.empty:
        report["warnings"].append("Empty dataset")
        report["quality_score"] = 0
        return report

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # ----- Duplicates -----
    dup_mask = df.duplicated(subset=["date", "store_id", "sku_id"], keep=False)
    n_dups = int(dup_mask.sum())
    report["issues"]["duplicates"] = {
        "count": n_dups,
        "pct": round(n_dups / len(df) * 100, 2),
    }
    if n_dups:
        report["warnings"].append(f"{n_dups} duplicate rows found")

    # ----- Invalid values -----
    invalid_reasons = []
    if (df["sales_units"] < 0).any():
        invalid_reasons.append("negative sales_units")
    if "price" in df.columns and (df["price"] < 0).any():
        invalid_reasons.append("negative price")
    if "inventory_level" in df.columns and (df["inventory_level"] < 0).any():
        invalid_reasons.append("negative inventory_level")
    if "promotion_flag" in df.columns and not df["promotion_flag"].isin([0, 1]).all():
        invalid_reasons.append("promotion_flag not in {0, 1}")

    report["issues"]["invalid"] = {
        "count": len(invalid_reasons),
        "reasons": invalid_reasons,
    }
    for r in invalid_reasons:
        report["warnings"].append(f"Invalid data: {r}")

    # ----- Date issues -----
    today = pd.Timestamp.now().normalize()
    future_mask = df["date"] > today
    n_future = int(future_mask.sum())
    report["issues"]["future_dates"] = {
        "count": n_future,
        "sample": df[future_mask]["date"].dt.date.astype(str).unique().tolist()[:5],
    }
    if n_future:
        report["warnings"].append(f"{n_future} rows with future dates")

    # ----- Gaps -----
    gaps_summary = []
    for (store, sku), grp in df.groupby(["store_id", "sku_id"]):
        dates = grp["date"].sort_values().unique()
        if len(dates) < 2:
            continue
        span_days = (dates[-1] - dates[0]) / np.timedelta64(1, "D")
        if span_days < 2:
            continue
        expected = int(span_days) + 1
        missing = expected - len(dates)
        if missing > 0:
            gaps_summary.append({
                "store_id": int(store),
                "sku_id": int(sku),
                "missing_days": missing,
                "range_days": expected,
            })

    report["issues"]["gaps"] = {
        "count": len(gaps_summary),
        "total_missing_days": sum(g["missing_days"] for g in gaps_summary),
        "sample": gaps_summary[:5],
    }
    if gaps_summary:
        report["warnings"].append(
            f"{len(gaps_summary)} store-SKU combos have missing dates"
        )

    # ----- Anomalies -----
    anomalies = []
    for (store, sku), grp in df.groupby(["store_id", "sku_id"]):
        if len(grp) < 5:
            continue
        mean = grp["sales_units"].mean()
        std = grp["sales_units"].std()
        if std and std > 0:
            z = (grp["sales_units"] - mean) / std
            outliers = grp[abs(z) > 3]
            if len(outliers):
                anomalies.append({
                    "store_id": int(store),
                    "sku_id": int(sku),
                    "outlier_count": int(len(outliers)),
                })

    report["issues"]["anomalies"] = {
        "count": len(anomalies),
        "details": anomalies[:5],
    }
    if anomalies:
        report["warnings"].append(
            f"{len(anomalies)} store-SKU combos have sales anomalies"
        )

    # ----- Quality score -----
    score = 100
    score -= min(30, report["issues"]["duplicates"]["pct"])
    score -= min(20, report["issues"]["invalid"]["count"] * 5)
    score -= min(20, report["issues"]["future_dates"]["count"] // 10)
    score -= min(20, report["issues"]["gaps"]["count"])
    score -= min(20, report["issues"]["anomalies"]["count"])
    report["quality_score"] = max(0, int(score))

    return report