"""FastAPI application for retail sales forecasting with POS upload."""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd

from config.settings import config
from config.logging_config import logger
from api.auth import get_current_user, require_role
from api.data_quality import analyze_quality
from api import retrain as retrain_module


# ============================================================
# APP INITIALIZATION
# ============================================================
app = FastAPI(
    title="Retail Sales Forecasting API",
    version="1.0.0",
    description="ML-powered sales forecasting with POS data upload",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# GLOBAL STATE
# ============================================================
model_bundle = None
model_loaded_time = None
history_df = None

UPLOAD_PATH = config.data.raw_data_path / "pos_uploads.parquet"
EXPECTED_COLUMNS = [
    "date", "store_id", "sku_id", "sales_units",
    "price", "promotion_flag", "inventory_level",
]


# ============================================================
# STARTUP
# ============================================================
@app.on_event("startup")
async def startup():
    global model_bundle, model_loaded_time, history_df
    logger.info("Starting API...")

    latest = config.data.models_path / "latest.pkl"
    if latest.exists():
        model_bundle = joblib.load(latest)
        model_loaded_time = datetime.now()
        logger.info(f"Model loaded from {latest}")
    else:
        logger.warning("No model found. Train first: python -m src.models.train")

    clean_path = config.data.processed_data_path / "sales_clean.parquet"
    if clean_path.exists():
        history_df = pd.read_parquet(clean_path)
        logger.info(f"History loaded: {len(history_df):,} rows")
    else:
        logger.warning(f"No history at {clean_path}. Run training first.")


# ============================================================
# ROOT / HEALTH / MODEL INFO
# ============================================================
@app.get("/")
async def root():
    return {
        "name": "Retail Sales Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "forecast": "POST /forecast",
            "upload_manual": "POST /upload/manual",
            "upload_file": "POST /upload/csv",
            "upload_summary": "GET /upload/summary",
            "retrain_status": "GET /retrain/status",
            "whoami": "GET /auth/whoami",
        },
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_bundle is not None,
        "history_loaded": history_df is not None,
        "environment": config.environment,
    }


@app.get("/model/info")
async def model_info(user: dict = Depends(get_current_user)):
    if model_bundle is None:
        raise HTTPException(503, "No model loaded")
    return {
        "model_type": "lightgbm",
        "n_features": len(model_bundle["features"]),
        "features": model_bundle["features"],
        "loaded_at": model_loaded_time.isoformat() if model_loaded_time else None,
    }


@app.get("/auth/whoami")
async def whoami(user: dict = Depends(get_current_user)):
    return user


# ============================================================
# FORECAST
# ============================================================
@app.post("/forecast")
async def forecast(
    store_id: int,
    sku_id: int,
    horizon: int = 7,
    user: dict = Depends(get_current_user),
):
    """Generate a varied forecast for a store-SKU using recent history."""
    if model_bundle is None:
        raise HTTPException(503, "Model not loaded. Train first.")
    if history_df is None:
        raise HTTPException(503, "No historical data. Train first.")

    model = model_bundle["model"]
    features = model_bundle["features"]

    sku_hist = history_df[
        (history_df["store_id"] == store_id) & (history_df["sku_id"] == sku_id)
    ].sort_values("date")

    if sku_hist.empty:
        raise HTTPException(404, f"No history for store={store_id}, sku={sku_id}")

    recent = sku_hist.tail(28)
    base_mean = float(recent["sales_units"].mean())
    base_std = float(recent["sales_units"].std() or 1.0)
    base_price = float(recent["price"].mean())
    base_inventory = float(recent["inventory_level"].iloc[-1])

    dow_factor = (
        recent.groupby(recent["date"].dt.dayofweek)["sales_units"].mean()
        / (base_mean or 1.0)
    )
    dow_factor = dow_factor.fillna(1.0).clip(0.5, 2.0)

    recent_sales = list(recent["sales_units"].astype(float).values)
    if len(recent_sales) < 30:
        recent_sales = [base_mean] * (30 - len(recent_sales)) + recent_sales

    predictions = []
    start_date = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    for i in range(horizon):
        forecast_date = start_date + pd.Timedelta(days=i)
        dow = forecast_date.dayofweek

        row = _build_feature_row(
            features=features,
            forecast_date=forecast_date,
            recent_sales=recent_sales,
            base_price=base_price,
            base_inventory=base_inventory,
            base_mean=base_mean,
        )

        X = pd.DataFrame([row])
        pred = float(model.predict(X)[0])
        pred = pred * float(dow_factor.get(dow, 1.0))
        pred = pred + np.random.normal(0, base_std * 0.10)
        pred = max(0.0, round(pred, 2))

        predictions.append({
            "date": forecast_date.date().isoformat(),
            "forecast": pred,
        })

        recent_sales.append(pred)
        recent_sales = recent_sales[-30:]

    return {
        "store_id": store_id,
        "sku_id": sku_id,
        "horizon": horizon,
        "predictions": predictions,
        "baseline_daily_sales": round(base_mean, 2),
        "model_type": "lightgbm",
        "requested_by": user["user"],
    }


# ============================================================
# HELPERS
# ============================================================
def _build_feature_row(
    features, forecast_date, recent_sales, base_price, base_inventory, base_mean
):
    row = {f: 0.0 for f in features}

    def safe_mean(lst, n):
        vals = lst[-n:] if len(lst) >= n else lst
        return float(np.mean(vals)) if vals else 0.0

    def safe_std(lst, n):
        vals = lst[-n:] if len(lst) >= n else lst
        return float(np.std(vals)) if vals else 0.0

    for lag in [1, 7, 14, 28, 365]:
        key = f"sales_lag_{lag}"
        if key in row:
            if len(recent_sales) >= lag:
                row[key] = recent_sales[-lag]
            else:
                row[key] = safe_mean(recent_sales, len(recent_sales))

    for w in [7, 28]:
        if f"sales_roll_mean_{w}" in row:
            row[f"sales_roll_mean_{w}"] = safe_mean(recent_sales, w)
        if f"sales_roll_std_{w}" in row:
            row[f"sales_roll_std_{w}"] = safe_std(recent_sales, w)

    if "price_lag_1" in row:
        row["price_lag_1"] = base_price
    if "price" in row:
        row["price"] = base_price
    if "promotion_flag" in row:
        row["promotion_flag"] = 0
    if "inventory_level" in row:
        row["inventory_level"] = base_inventory

    if "day_of_week" in row:
        row["day_of_week"] = forecast_date.dayofweek
    if "month" in row:
        row["month"] = forecast_date.month
    if "quarter" in row:
        row["quarter"] = forecast_date.quarter
    if "year" in row:
        row["year"] = forecast_date.year
    if "day_of_year" in row:
        row["day_of_year"] = forecast_date.dayofyear
    if "is_weekend" in row:
        row["is_weekend"] = int(forecast_date.dayofweek in [5, 6])

    if "month_sin" in row:
        row["month_sin"] = np.sin(2 * np.pi * forecast_date.month / 12)
    if "month_cos" in row:
        row["month_cos"] = np.cos(2 * np.pi * forecast_date.month / 12)
    if "dow_sin" in row:
        row["dow_sin"] = np.sin(2 * np.pi * forecast_date.dayofweek / 7)
    if "dow_cos" in row:
        row["dow_cos"] = np.cos(2 * np.pi * forecast_date.dayofweek / 7)

    if "days_since_promo" in row:
        row["days_since_promo"] = 30
    if "is_stockout" in row:
        row["is_stockout"] = 0
    if "store_id" in row:
        row["store_id"] = 0
    if "sku_id" in row:
        row["sku_id"] = 0

    return row


# ============================================================
# POS DATA UPLOAD
# ============================================================
def _validate_upload(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize uploaded POS data."""
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    aliases = {
        "qty": "sales_units",
        "quantity": "sales_units",
        "units_sold": "sales_units",
        "sales": "sales_units",
        "store": "store_id",
        "shop_id": "store_id",
        "sku": "sku_id",
        "product_id": "sku_id",
        "inventory": "inventory_level",
        "on_hand": "inventory_level",
        "promo": "promotion_flag",
        "is_promo": "promotion_flag",
        "unit_price": "price",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    required = ["date", "store_id", "sku_id", "sales_units"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"Missing required columns: {missing}. Found: {list(df.columns)}",
        )

    if "price" not in df.columns:
        df["price"] = 0.0
    if "promotion_flag" not in df.columns:
        df["promotion_flag"] = 0
    if "inventory_level" not in df.columns:
        df["inventory_level"] = 0

    try:
        df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise HTTPException(400, f"Invalid date format: {e}")

    for col in ["store_id", "sku_id", "sales_units", "promotion_flag", "inventory_level"]:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception as e:
            raise HTTPException(400, f"Column '{col}' must be numeric: {e}")

    for col in ["store_id", "sku_id", "promotion_flag", "inventory_level"]:
        df[col] = df[col].astype(int)

    df["price"] = df["price"].astype(float)
    df["sales_units"] = df["sales_units"].astype(int)

    before = len(df)
    df = df[df["sales_units"] >= 0]
    df = df.dropna(subset=["date", "store_id", "sku_id"])
    dropped = before - len(df)
    if dropped:
        logger.warning(f"Dropped {dropped} invalid rows")

    df = df[EXPECTED_COLUMNS]

    return df


def _append_to_store(df: pd.DataFrame) -> int:
    UPLOAD_PATH.parent.mkdir(parents=True, exist_ok=True)

    if UPLOAD_PATH.exists():
        existing = pd.read_parquet(UPLOAD_PATH)
        existing["date"] = pd.to_datetime(existing["date"])
        combined = pd.concat([existing, df], ignore_index=True)
    else:
        combined = df.copy()

    combined = combined.drop_duplicates(
        subset=["date", "store_id", "sku_id"], keep="last"
    ).reset_index(drop=True)

    combined.to_parquet(UPLOAD_PATH, index=False)
    logger.info(f"POS store now has {len(combined):,} rows")
    return len(df)


@app.post("/upload/manual")
async def upload_manual(
    date: str = Form(...),
    store_id: int = Form(...),
    sku_id: int = Form(...),
    sales_units: int = Form(...),
    price: float = Form(0.0),
    promotion_flag: int = Form(0),
    inventory_level: int = Form(0),
    user: dict = Depends(require_role("admin", "analyst")),
):
    """Upload a single POS record (requires admin or analyst role)."""
    try:
        row = {
            "date": pd.to_datetime(date),
            "store_id": int(store_id),
            "sku_id": int(sku_id),
            "sales_units": int(sales_units),
            "price": float(price),
            "promotion_flag": int(promotion_flag),
            "inventory_level": int(inventory_level),
        }
    except Exception as e:
        raise HTTPException(400, f"Invalid input: {e}")

    df = pd.DataFrame([row])
    quality = analyze_quality(df)
    n = _append_to_store(df)
    retrain_status = retrain_module.increment_upload(n)

    logger.info(f"User '{user['user']}' uploaded 1 manual row")

    return {
        "status": "success",
        "rows_received": n,
        "message": f"Added 1 record for store={store_id}, sku={sku_id} on {date}",
        "uploaded_by": user["user"],
        "quality": quality,
        "retrain": retrain_status,
    }


@app.post("/upload/csv")
async def upload_csv(
    file: UploadFile = File(...),
    user: dict = Depends(require_role("admin", "analyst")),
):
    """Upload a CSV or XLSX file with POS data."""
    filename = file.filename.lower()

    if filename.endswith(".csv") or filename.endswith(".txt"):
        kind = "csv"
    elif filename.endswith(".xlsx") or filename.endswith(".xls"):
        kind = "excel"
    else:
        raise HTTPException(400, "Only .csv, .xlsx, or .xls files are supported")

    contents = await file.read()
    if not contents:
        raise HTTPException(400, "Uploaded file is empty")

    try:
        if kind == "csv":
            df = pd.read_csv(BytesIO(contents))
        else:
            df = pd.read_excel(BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Cannot parse {kind}: {e}")

    if df.empty:
        raise HTTPException(400, "Uploaded file contains no rows")

    logger.info(f"User '{user['user']}' uploaded {filename} ({len(df)} rows)")

    df = _validate_upload(df)
    if df.empty:
        raise HTTPException(400, "No valid rows after validation")

    quality = analyze_quality(df)
    n = _append_to_store(df)
    retrain_status = retrain_module.increment_upload(n)

    return {
        "status": "success",
        "filename": file.filename,
        "file_type": kind,
        "rows_received": n,
        "rows_valid": len(df),
        "columns": list(df.columns),
        "date_range": {
            "min": df["date"].min().date().isoformat(),
            "max": df["date"].max().date().isoformat(),
        },
        "stores": sorted(df["store_id"].unique().tolist())[:20],
        "skus": sorted(df["sku_id"].unique().tolist())[:20],
        "uploaded_by": user["user"],
        "quality": quality,
        "retrain": retrain_status,
    }


@app.get("/upload/summary")
async def upload_summary(user: dict = Depends(get_current_user)):
    if not UPLOAD_PATH.exists():
        return {"total_rows": 0, "message": "No uploads yet"}

    df = pd.read_parquet(UPLOAD_PATH)
    df["date"] = pd.to_datetime(df["date"])

    return {
        "total_rows": len(df),
        "date_range": {
            "min": df["date"].min().date().isoformat(),
            "max": df["date"].max().date().isoformat(),
        },
        "n_stores": int(df["store_id"].nunique()),
        "n_skus": int(df["sku_id"].nunique()),
        "total_sales_units": int(df["sales_units"].sum()),
        "last_updated": pd.Timestamp(
            UPLOAD_PATH.stat().st_mtime, unit="s"
        ).isoformat(),
    }


@app.get("/upload/preview")
async def upload_preview(
    limit: int = 20, user: dict = Depends(get_current_user)
):
    if not UPLOAD_PATH.exists():
        return {"rows": [], "message": "No uploads yet"}

    df = pd.read_parquet(UPLOAD_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df = df.sort_values("date", ascending=False).head(limit)

    return {"rows": df.to_dict(orient="records"), "count": len(df)}


@app.delete("/upload/clear")
async def clear_uploads(user: dict = Depends(require_role("admin"))):
    if UPLOAD_PATH.exists():
        UPLOAD_PATH.unlink()
        logger.warning(f"User '{user['user']}' cleared all POS uploads")
    return {"status": "cleared", "by": user["user"]}


# ============================================================
# RETRAIN & MONITORING
# ============================================================
@app.get("/retrain/status")
async def retrain_status(user: dict = Depends(get_current_user)):
    return retrain_module.get_status()


@app.post("/retrain/trigger")
async def retrain_trigger(user: dict = Depends(require_role("admin"))):
    status = retrain_module.get_status()
    retrain_module._trigger_retrain({"upload_count": status["upload_count"]})
    return {"status": "triggered", "by": user["user"]}


@app.post("/retrain/reset")
async def retrain_reset(user: dict = Depends(require_role("admin"))):
    return retrain_module.reset_counter()


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=config.deployment.api_host,
        port=config.deployment.api_port,
        reload=config.debug,
    )