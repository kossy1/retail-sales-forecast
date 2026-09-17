"""Global configuration using Pydantic v2 Settings."""

from pathlib import Path
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_ROOT = PROJECT_ROOT / "data"


class DataConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    raw_data_path: Path = DATA_ROOT / "raw"
    interim_data_path: Path = DATA_ROOT / "interim"
    processed_data_path: Path = DATA_ROOT / "processed"
    models_path: Path = PROJECT_ROOT / "models" / "saved"

    source_type: str = Field("local", alias="DATA_SOURCE_TYPE")
    db_connection_string: Optional[str] = None

    start_date: str = "2021-01-01"
    end_date: str = "2023-12-31"
    sample_size: Optional[int] = None


class FeatureConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    group_by_columns: List[str] = ["store_id", "sku_id"]
    lag_days: List[int] = [1, 7, 14, 28, 365]
    rolling_windows: List[int] = [7, 28]
    rolling_stats: List[str] = ["mean", "std"]
    categorical_columns: List[str] = ["store_id", "sku_id"]


class ModelConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_type: str = Field("lightgbm", alias="MODEL_TYPE")
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42

    lgbm_params: Dict[str, Any] = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_jobs": -1,
        "random_state": 42,
        "verbose": -1,
    }


class DeploymentConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    enable_cors: bool = True


class LoggingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field("INFO", alias="LOG_LEVEL")
    log_file_path: Path = PROJECT_ROOT / "logs" / "retail_forecast.log"


class AuthConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    enabled: bool = Field(True, alias="AUTH_ENABLED")
    api_keys_file: Path = PROJECT_ROOT / "config" / "api_keys.json"


class RetrainConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    enabled: bool = Field(True, alias="AUTO_RETRAIN_ENABLED")
    threshold_uploads: int = Field(100, alias="AUTO_RETRAIN_THRESHOLD")
    counter_file: Path = PROJECT_ROOT / "data" / "upload_counter.json"


class GlobalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    project_name: str = "Retail Sales Forecasting"
    environment: str = Field("development", alias="ENVIRONMENT")
    debug: bool = Field(False, alias="DEBUG")

    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    model: ModelConfig = ModelConfig()
    deployment: DeploymentConfig = DeploymentConfig()
    logging: LoggingConfig = LoggingConfig()
    auth: AuthConfig = AuthConfig()
    retrain: RetrainConfig = RetrainConfig()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self):
        for d in [
            self.data.raw_data_path,
            self.data.interim_data_path,
            self.data.processed_data_path,
            self.data.models_path,
            self.logging.log_file_path.parent,
        ]:
            d.mkdir(parents=True, exist_ok=True)


config = GlobalConfig()