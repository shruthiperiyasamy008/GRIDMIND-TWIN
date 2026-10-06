"""Energy data processing and analytics services.

Preprocessing (pandas / numpy based):
- validate readings
- sort timestamps
- aggregate hourly
- rolling averages
- baseline consumption
- peak demand
- daily energy
- zone-level energy
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from app.config import settings


def frame_from_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, de-duplicate and sort; interpolate missing values."""
    if df.empty:
        return df
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "zone_id"]).sort_values(
        "timestamp"
    ).reset_index(drop=True)

    numeric_cols = [
        "power_kw", "energy_kwh", "temperature", "occupancy",
        "hvac_load", "lighting_load", "pump_load", "other_load",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Validate: non-negative power
    for col in ["power_kw", "energy_kwh", "hvac_load", "lighting_load", "pump_load", "other_load"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # Missing values -> linear interpolation where feasible
    if "power_kw" in df.columns:
        df["power_kw"] = df["power_kw"].interpolate(method="linear").fillna(0.0)
        df["energy_kwh"] = df["energy_kwh"].interpolate(method="linear").fillna(0.0)
        for col in ["temperature", "occupancy", "hvac_load", "lighting_load", "pump_load", "other_load"]:
            if col in df.columns:
                df[col] = df[col].interpolate(method="linear").fillna(df[col].median() if not df[col].isna().all() else 0.0)

    return df


def aggregate_building_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate zone rows into building-level hourly rows."""
    if df.empty:
        return df
    df = df.set_index("timestamp")
    cols = {
        "power_kw": "sum", "energy_kwh": "sum", "hvac_load": "sum",
        "lighting_load": "sum", "pump_load": "sum", "other_load": "sum",
        "temperature": "mean", "occupancy": "mean",
    }
    cols = {k: v for k, v in cols.items() if k in df.columns}
    agg = df[list(cols.keys())].resample("h").agg(cols)
    return agg.reset_index()


def zone_energy_summary(df: pd.DataFrame) -> list[dict]:
    """Energy consumed per zone (kWh)."""
    if df.empty:
        return []
    g = df.groupby("zone_id").agg(
        energy_kwh=("energy_kwh", "sum"),
        mean_power=("power_kw", "mean"),
        peak_power=("power_kw", "max"),
    ).reset_index()
    return g.to_dict(orient="records")


def rolling_averages(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["rolling_avg_kw"] = out["power_kw"].rolling(window=window, min_periods=1).mean()
    return out


def baseline_consumption(df: pd.DataFrame) -> float:
    """Typical hourly baseline (median historical consumption)."""
    if df.empty:
        return 0.0
    return float(df["power_kw"].median())


def peak_demand(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"peak_kw": 0.0, "peak_time": None}
    row = df.loc[df["power_kw"].idxmax()]
    return {
        "peak_kw": float(row["power_kw"]),
        "peak_time": str(row["timestamp"]),
    }


def daily_energy_series(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    g = df.groupby("date").agg(
        energy_kwh=("energy_kwh", "sum"),
        peak_kw=("power_kw", "max"),
    ).reset_index()
    return [
        {"date": str(r["date"]), "energy_kwh": round(float(r["energy_kwh"]), 2),
         "peak_kw": round(float(r["peak_kw"]), 2)}
        for r in g.to_dict(orient="records")
    ]


def cost_and_carbon(energy_kwh: float) -> dict:
    tariff = settings.ENERGY_TARIFF_PER_KWH
    factor = settings.CARBON_FACTOR_KG_PER_KWH
    return {
        "energy_kwh": round(energy_kwh, 2),
        "cost_inr": round(energy_kwh * tariff, 2),
        "carbon_kg": round(energy_kwh * factor, 2),
        "tariff_per_kwh": tariff,
        "carbon_factor_kg_per_kwh": factor,
        "note": "ESTIMATED using configurable tariff & emissions factors",
    }


def hourly_profile(df: pd.DataFrame) -> list[dict]:
    """Mean consumption by hour of day."""
    if df.empty:
        return []
    df = df.copy()
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    g = df.groupby("hour").agg(mean_kw=("power_kw", "mean")).reset_index()
    return [{"hour": int(r["hour"]), "mean_kw": round(float(r["mean_kw"]), 2)}
            for r in g.to_dict(orient="records")]


def summarize(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"status": "EMPTY DATASET"}
    total_energy = float(df["energy_kwh"].sum())
    peak = peak_demand(df)
    p95 = float(np.percentile(df["power_kw"], 95))
    cc = cost_and_carbon(total_energy)
    return {
        "source": settings.DATA_SOURCE_LABEL,
        "rows": int(len(df)),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "total_energy_kwh": round(total_energy, 2),
        "mean_power_kw": round(float(df["power_kw"].mean()), 2),
        "peak_kw": peak["peak_kw"],
        "peak_time": peak["peak_time"],
        "p95_kw": round(p95, 2),
        "estimated_cost_inr": cc["cost_inr"],
        "estimated_carbon_kg": cc["carbon_kg"],
    }


def resample_series(df: pd.DataFrame, granularity: str) -> list[dict]:
    """Resample building-level hourly rows to daily / weekly / monthly."""
    if df.empty:
        return []
    rules = {"daily": "D", "weekly": "W", "monthly": "MS"}
    rule = rules.get(granularity, "D")
    df = df.set_index("timestamp")
    agg_cols = {
        "power_kw": "sum", "energy_kwh": "sum", "hvac_load": "sum",
        "lighting_load": "sum", "temperature": "mean",
    }
    agg_cols = {k: v for k, v in agg_cols.items() if k in df.columns}
    g = df[list(agg_cols.keys())].resample(rule).agg(agg_cols).reset_index()
    out = []
    for r in g.to_dict(orient="records"):
        label = {"daily": "date", "weekly": "week", "monthly": "month"}.get(granularity, "date")
        out.append({label: str(r["timestamp"].date()), **{
            "energy_kwh": round(float(r["energy_kwh"]), 2) if "energy_kwh" in r else round(float(r["power_kw"]), 2),
            "peak_kw": round(float(r["power_kw"]), 2) if "power_kw" in r else 0.0,
        }})
    return out


def equipment_split(building_df: pd.DataFrame) -> list[dict]:
    """Energy by load type (equipment category proxy)."""
    if building_df.empty:
        return []
    total = float(building_df["energy_kwh"].sum())
    if total <= 0:
        return []
    parts = [
        ("HVAC", float(building_df["hvac_load"].sum())),
        ("Lighting", float(building_df["lighting_load"].sum())),
        ("Pumps", float(building_df["pump_load"].sum())),
        ("Other Loads", float(building_df["other_load"].sum())),
    ]
    return [{"name": name, "energy_kwh": round(v, 2),
             "share_pct": round(v / total * 100, 1)} for name, v in parts]