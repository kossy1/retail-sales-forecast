"""Auto-retrain trigger when enough uploads accumulate."""

import json
import threading
from datetime import datetime
from typing import Dict, Any
from config.settings import config
from config.logging_config import logger


_lock = threading.Lock()


def _load_counter() -> Dict[str, Any]:
    path = config.retrain.counter_file
    if not path.exists():
        return {"upload_count": 0, "last_train": None, "history": []}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"upload_count": 0, "last_train": None, "history": []}


def _save_counter(data: Dict[str, Any]):
    path = config.retrain.counter_file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def increment_upload(rows: int) -> Dict[str, Any]:
    """Increment upload counter. Trigger retrain if threshold hit."""
    with _lock:
        data = _load_counter()
        data["upload_count"] = data.get("upload_count", 0) + rows
        _save_counter(data)

        triggered = False
        if (
            config.retrain.enabled
            and data["upload_count"] >= config.retrain.threshold_uploads
        ):
            triggered = _trigger_retrain(data)

        return {
            "upload_count": data["upload_count"],
            "threshold": config.retrain.threshold_uploads,
            "triggered": triggered,
        }


def _trigger_retrain(data: Dict[str, Any]) -> bool:
    """Run training in a background thread."""
    logger.info("Auto-retrain threshold reached — starting training...")

    def _train():
        try:
            from src.data_pipeline.load_data import load_and_clean_data
            from src.features.build_features import build_features
            from src.models.train import ModelTrainer

            df = load_and_clean_data()
            df, feats = build_features(df)
            trainer = ModelTrainer()
            trainer.train(df, feats)

            data["last_train"] = datetime.now().isoformat()
            data["upload_count"] = 0
            data.setdefault("history", []).append({
                "trained_at": datetime.now().isoformat(),
                "reason": "auto-threshold",
            })
            _save_counter(data)
            logger.info("Auto-retrain completed")
        except Exception as e:
            logger.error(f"Auto-retrain failed: {e}")

    threading.Thread(target=_train, daemon=True).start()
    return True


def get_status() -> Dict[str, Any]:
    data = _load_counter()
    threshold = config.retrain.threshold_uploads or 1
    return {
        "enabled": config.retrain.enabled,
        "upload_count": data.get("upload_count", 0),
        "threshold": threshold,
        "last_train": data.get("last_train"),
        "progress_pct": round(
            min(100, data.get("upload_count", 0) / threshold * 100), 1
        ),
        "history": data.get("history", [])[-5:],
    }


def reset_counter() -> Dict[str, Any]:
    data = _load_counter()
    data["upload_count"] = 0
    _save_counter(data)
    return {"status": "reset", "upload_count": 0}