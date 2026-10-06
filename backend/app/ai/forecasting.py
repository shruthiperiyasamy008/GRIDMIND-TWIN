"""Energy forecasting using scikit-learn.

Feature engineering:
- hour, day_of_week, weekend flag
- temperature, occupancy
- lag features (previous consumption)
- rolling average

Model: RandomForestRegressor trained on the simulated historical dataset.
Returns next 24 hours of predicted power with a confidence band.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FORECAST_HORIZON = 24


def _features(df: pd.DataFrame, target="power_kw") -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    out = pd.DataFrame(index=df.index)
    ts = pd.to_datetime(df["timestamp"])
    out["hour"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    out["temperature"] = df["temperature"].fillna(25.0)
    out["occupancy"] = df["occupancy"].fillna(0.0)
    out["t_minus_1"] = df[target].shift(1)
    out["t_minus_2"] = df[target].shift(2)
    out["t_minus_24"] = df[target].shift(24)
    out["rolling_avg_24"] = df[target].rolling(24, min_periods=1).mean().shift(1)
    out["tariff_hour"] = ((ts.dt.hour >= 8) & (ts.dt.hour <= 22)).astype(int)
    return out


class EnergyForecaster:
    """Next-24h forecaster trained on historical consumption."""

    def __init__(self, model=None):
        self.model = model or RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=1,
        )
        self.feature_cols = [
            "hour", "day_of_week", "weekend", "temperature", "occupancy",
            "t_minus_1", "t_minus_2", "t_minus_24", "rolling_avg_24", "tariff_hour",
        ]

    def train(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 200:
            raise ValueError("Not enough historical data to train a forecast model.")
        feats = _features(df).dropna()
        if len(feats) < 100:
            raise ValueError("Not enough valid feature rows to train the model.")
        y = df.loc[feats.index, "power_kw"]
        y = pd.to_numeric(y, errors="coerce").dropna()
        feats = feats.loc[y.index]
        self.model.fit(feats[self.feature_cols], y)
        residuals = y - self.model.predict(feats[self.feature_cols])
        self.mae = float(np.mean(np.abs(residuals)))
        self.rmse = float(np.sqrt(np.mean(residuals ** 2)))
        return {"samples": int(len(y)), "mae_kw": round(self.mae, 2),
                "rmse_kw": round(self.rmse, 2), "model": type(self.model).__name__}

    def forecast(self, df: pd.DataFrame, horizon: int = FORECAST_HORIZON) -> dict:
        if self.model is None or not hasattr(self.model, "predict"):
            raise ValueError("Forecast model must be trained first.")

        df = df.sort_values("timestamp").reset_index(drop=True)
        last_ts = df["timestamp"].max()
        feats = _features(df)
        recent = df.copy()

        preds = []
        for step in range(horizon):
            ts = last_ts + timedelta(hours=step + 1)
            last_row = recent.tail(1)
            f = {
                "hour": ts.hour,
                "day_of_week": ts.weekday(),
                "weekend": int(ts.weekday() >= 5),
                "temperature": float(last_row["temperature"].iloc[0]) if not last_row.empty else 25.0,
                "occupancy": float(np.clip(
                    _profile_from_activity(ts) * 100.0, 0, 100)),
                "t_minus_1": float(last_row["power_kw"].iloc[0]) if not last_row.empty else 0,
                "t_minus_2": float(recent["power_kw"].iloc[-2]) if len(recent) >= 2 else 0,
                "t_minus_24": float(recent["power_kw"].iloc[-24]) if len(recent) >= 24 else float(last_row["power_kw"].iloc[0]) if not last_row.empty else 0,
                "rolling_avg_24": float(recent["power_kw"].tail(24).mean()) if len(recent) >= 1 else 0,
                "tariff_hour": int(8 <= ts.hour <= 22),
            }
            pred = float(self.model.predict(pd.DataFrame([f])[self.feature_cols])[0])
            pred = max(0.0, pred)

            # concentration band grows with horizon
            band = self.mae * (1 + 0.08 * step)
            preds.append({
                "timestamp": ts.isoformat(),
                "predicted_power_kw": round(pred, 2),
                "lower_kw": round(max(0, pred - band), 2),
                "upper_kw": round(pred + band, 2),
                "is_forecast": True,
            })
            recent = pd.concat([recent, pd.DataFrame([{
                "timestamp": ts, "power_kw": pred,
                "temperature": f["temperature"], "occupancy": f["occupancy"],
                **{c: last_row[c].iloc[0] if not last_row.empty else 0 for c in
                   ["hvac_load", "lighting_load", "pump_load", "other_load"]},
            }])], ignore_index=True)

        peak = max(preds, key=lambda p: p["predicted_power_kw"])
        return {
            "horizon_hours": horizon,
            "data_source": "PREDICTION",
            "forecast": preds,
            "predicted_peak_kw": round(peak["predicted_power_kw"], 2),
            "predicted_peak_time": peak["timestamp"],
            "model_metrics": {"mae_kw": round(self.mae, 2), "rmse_kw": round(self.rmse, 2)},
            "confidence": round(max(0, 1 - self.mae / max(1e-6, float(df["power_kw"].median()))) * 100, 1),
        }


def _profile_from_activity(ts: datetime) -> float:
    """Occupancy proxy from hour + weekday (daylight schedule)."""
    hour = ts.hour
    if ts.weekday() >= 5:
        profile = [0.08, 0.07, 0.06, 0.06, 0.06, 0.07, 0.10, 0.18, 0.28, 0.34,
                   0.36, 0.40, 0.42, 0.44, 0.42, 0.40, 0.36, 0.30, 0.24, 0.20,
                   0.16, 0.13, 0.11, 0.10]
    else:
        profile = [0.05, 0.05, 0.05, 0.05, 0.06, 0.10, 0.32, 0.60, 0.80, 0.90,
                   0.94, 0.96, 0.97, 0.95, 0.90, 0.82, 0.72, 0.60, 0.48, 0.40,
                   0.32, 0.24, 0.15, 0.08]
    return profile[hour]


class ForecastService:
    """Caches a trained model for the current dataset."""

    def __init__(self):
        self.model = None
        self.last_frame = None

    def get_forecast(self, df: pd.DataFrame) -> dict:
        if df.empty:
            raise ValueError("EMPTY DATASET: no energy readings available for forecasting.")
        # Retrain if the data changed (cheap for demo sizes)
        df = df.sort_values("timestamp").reset_index(drop=True)
        if (
            self.last_frame is None
            or len(df) != len(self.last_frame)
            or (df["timestamp"].iloc[-1] != self.last_frame["timestamp"].iloc[-1])
        ):
            forecaster = EnergyForecaster()
            metrics = forecaster.train(df)
            self.model = forecaster
            self.last_frame = df.copy()
        return self.model.forecast(df)


forecast_service = ForecastService()
