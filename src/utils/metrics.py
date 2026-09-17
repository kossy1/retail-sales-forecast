"""
Custom evaluation metrics for retail forecasting.
Focuses on business-relevant metrics like WAPE and bias.
"""

import numpy as np
import pandas as pd
from typing import Union, Optional, List, Dict
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_wape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    weights: Optional[Union[np.ndarray, pd.Series]] = None
) -> float:
    """
    Weighted Absolute Percentage Error (WAPE).
    Also known as MAD/Mean ratio.
    
    For retail, WAPE is preferred over MAPE because it:
    - Weighs high-volume items more heavily
    - Avoids division by zero issues
    - Better reflects total revenue impact
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        weights: Optional weights (e.g., price or volume)
    
    Returns:
        WAPE as a percentage
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Use weights if provided, otherwise unweighted
    if weights is not None:
        weights = np.array(weights).flatten()
        if len(weights) != len(y_true):
            raise ValueError("weights must have the same length as y_true")
        wape = np.sum(weights * np.abs(y_true - y_pred)) / np.sum(weights * np.abs(y_true))
    else:
        wape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))
    
    # Handle edge case where sum of actuals is 0
    if np.sum(np.abs(y_true)) == 0:
        return np.nan
    
    return wape * 100


def calculate_bias(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Calculate forecast bias.
    
    Positive bias = over-forecasting
    Negative bias = under-forecasting
    
    For inventory management, consistent bias is more important
    than absolute error (can be corrected with safety stock).
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) == 0:
        return np.nan
    
    # Percentage bias
    with np.errstate(divide='ignore', invalid='ignore'):
        bias = np.mean((y_pred - y_true) / (y_true + 1)) * 100
    
    return bias


def calculate_mase(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_train: Optional[Union[np.ndarray, pd.Series]] = None,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error (MASE).
    
    MASE is scale-independent and works well for comparing
    across different SKUs with different sales volumes.
    
    MASE < 1 means the model beats the naive forecast.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    n = len(y_true)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate naive forecast error
    if y_train is not None:
        # Use training data for naive forecast
        y_train = np.array(y_train).flatten()
        if len(y_train) >= seasonality + 1:
            naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
            mae_naive = np.mean(naive_errors)
        else:
            mae_naive = np.mean(np.abs(y_train[1:] - y_train[:-1])) if len(y_train) > 1 else 1
    else:
        # Use in-sample naive forecast
        if n > seasonality:
            naive_errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
            mae_naive = np.mean(naive_errors)
        elif n > 1:
            naive_errors = np.abs(y_true[1:] - y_true[:-1])
            mae_naive = np.mean(naive_errors)
        else:
            mae_naive = 1
    
    if mae_naive == 0:
        return np.nan
    
    return mae / mae_naive


def calculate_smape(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series]
) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    Useful when actual values can be zero.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1, denominator)
    
    smape = np.mean(np.abs(y_true - y_pred) / denominator) * 100
    return smape


def calculate_forecast_value(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    price: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, float]:
    """
    Calculate business value of the forecast.
    
    This is a retail-specific metric that quantifies the
    financial impact of forecast accuracy.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    # If price is provided, calculate revenue impact
    if price is not None:
        price = np.array(price).flatten()
        if len(price) != len(y_true):
            raise ValueError("price must have the same length as y_true")
        
        actual_revenue = np.sum(y_true * price)
        predicted_revenue = np.sum(y_pred * price)
        revenue_error = np.sum(np.abs(y_true - y_pred) * price)
        revenue_error_pct = (revenue_error / actual_revenue) * 100 if actual_revenue > 0 else np.nan
    
    # Calculate over-forecasting cost (excess inventory)
    over_forecast = np.maximum(0, y_pred - y_true)
    over_forecast_pct = np.sum(over_forecast) / (np.sum(y_true) + 1) * 100
    
    # Calculate under-forecasting cost (lost sales)
    under_forecast = np.maximum(0, y_true - y_pred)
    under_forecast_pct = np.sum(under_forecast) / (np.sum(y_true) + 1) * 100
    
    metrics = {
        'over_forecast_pct': over_forecast_pct,
        'under_forecast_pct': under_forecast_pct,
    }
    
    if price is not None:
        metrics.update({
            'actual_revenue': actual_revenue,
            'predicted_revenue': predicted_revenue,
            'revenue_error': revenue_error,
            'revenue_error_pct': revenue_error_pct
        })
    
    return metrics


def calculate_all_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_train: Optional[Union[np.ndarray, pd.Series]] = None,
    weights: Optional[Union[np.ndarray, pd.Series]] = None,
    price: Optional[Union[np.ndarray, pd.Series]] = None
) -> Dict[str, float]:
    """
    Calculate a comprehensive set of metrics for retail forecasting.
    """
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'wape': calculate_wape(y_true, y_pred, weights),
        'bias': calculate_bias(y_true, y_pred),
        'smape': calculate_smape(y_true, y_pred),
        'mase': calculate_mase(y_true, y_pred, y_train),
    }
    
    # Add business value metrics
    if price is not None:
        business_metrics = calculate_forecast_value(y_true, y_pred, price)
        metrics.update(business_metrics)
    
    return metrics