"""Anomaly detection using IsolationForest (scikit-learn).

Features analysed: power_kw, hvac_load, lighting_load, occupancy, temperature.
Detects unusual consumption events and produces human-readable explanations.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

FEATURE_COLS = ["power_kw", "hvac_load", "lighting_load", "occupancy", "temperature"]


def _expected_reason(row: dict, expected: dict | None = None) -> str:
    """Heuristic explanation for an anomalous reading, attributing the cause
    to whichever load component deviates most from its typical level."""
    occ = row.get("occupancy", 50)
    power = row.get("power_kw", 0)
    hvac = row.get("hvac_load", 0)
    lighting = row.get("lighting_load", 0)
    other = row.get("other_load", 0)
    exp = expected or {}

    def deviation_of(value, expected_value):
        return (value - expected_value) / (expected_value if expected_value > 1 else 1.0)

    candidates = []
    if exp.get("hvac_exp") and hvac > (exp["hvac_exp"] * 1.25):
        candidates.append((deviation_of(hvac, exp["hvac_exp"]),
                           "High HVAC consumption" +
                           (" while zone occupancy is low." if occ < 30 else " — cooling load far above typical levels for this zone.")))
    if exp.get("lighting_exp") and lighting > (exp["lighting_exp"] * 1.5):
        candidates.append((deviation_of(lighting, exp["lighting_exp"]),
                           "Elevated lighting load" +
                           (" during low occupancy." if occ < 25 else " — lighting far above typical levels.")))
    if exp.get("other_exp") and other > (exp["other_exp"] * 2.0):
        candidates.append((deviation_of(other, exp["other_exp"]),
                           "Unusually high residual/miscellaneous load beyond typical usage."))
    if power and hvac == 0 and lighting == 0 and other == 0:
        pass
    if candidates:
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]
    if power > (exp.get("power_exp") or power) * 1.4:
        return "Consumption deviates sharply from expected zone behaviour."
    return "Consumption deviates sharply from expected zone behaviour."


class AnomalyDetector:
    def __init__(self, contamination=0.03, random_state=42):
        self.model = IsolationForest(
            n_estimators=80, contamination=contamination,
            random_state=random_state, n_jobs=1,
        )
        self.contamination = contamination

    def detect(self, df: pd.DataFrame, zone_names: dict[int, str] | None = None,
               lookback_days: int = 7) -> list[dict]:
        if df.empty or len(df) < 50:
            return []

        zone_names = zone_names or {}
        cutoff = df["timestamp"].max() - timedelta(days=lookback_days)
        window = df[df["timestamp"] >= cutoff]

        anomalies: list[dict] = []
        for zone_id, g in window.groupby("zone_id"):
            g = g.dropna(subset=FEATURE_COLS)
            if len(g) < 24:
                g_prev = df[df["zone_id"] == zone_id]
                g = g_prev.dropna(subset=FEATURE_COLS).tail(100)
                if len(g) < 24:
                    continue
            X = g[FEATURE_COLS].fillna(0).values
            pred = self.model.fit_predict(X)
            scores = self.model.decision_function(X)

            base_qt = np.percentile(g[FEATURE_COLS[0]].values, 50)
            # typical component levels for reasoning (median of zone window)
            def _med(col):
                if col not in g:
                    return 0.0
                return float(np.nan_to_num(np.median(g[col].values)))

            exp_hvac = _med("hvac_load")
            exp_lighting = _med("lighting_load")
            exp_other = _med("other_load")
            for i, (_, row) in enumerate(g.iterrows()):
                if pred[i] == -1:
                    obs = float(row["power_kw"])
                    # refined prediction of expected value using a kNN-ish median window
                    exp = float(row["power_kw"])
                    neighbours = window[
                        (window["zone_id"] == zone_id)
                        & (abs((window["hvac_load"] - row["hvac_load"])) < 40)
                        & (window.index != row.name)
                    ]["power_kw"]
                    if len(neighbours) >= 3:
                        exp = float(np.median(neighbours.tail(6)))
                    dev = (obs - exp) / (exp if exp > 1 else 1.0) * 100.0
                    severity = (
                        "CRITICAL" if dev > 60 else
                        "HIGH" if dev > 30 else
                        "MEDIUM" if dev > 10 else "LOW"
                    )
                    anomalies.append({
                        "zone_id": int(zone_id),
                        "zone_name": zone_names.get(int(zone_id), f"Zone {zone_id}"),
                        "timestamp": row["timestamp"].isoformat(),
                        "observed_power_kw": round(obs, 2),
                        "expected_power_kw": round(exp, 2),
                        "deviation_pct": round(dev, 1),
                        "severity": severity,
                        "isolation_score": round(float(scores[i]), 3),
                        "hvac_load_kw": round(float(row.get("hvac_load", 0)), 2),
                        "occupancy_pct": round(float(row.get("occupancy", 0)), 1),
                        "expected_components": {
                            "hvac_exp": round(float(exp_hvac), 2),
                            "lighting_exp": round(float(exp_lighting), 2),
                            "other_exp": round(float(exp_other), 2),
                        },
                        "possible_reason": _expected_reason(
                            {
                                "occupancy": row.get("occupancy", 50),
                                "hvac_load": row.get("hvac_load", 0),
                                "lighting_load": row.get("lighting_load", 0),
                                "other_load": row.get("other_load", 0),
                                "power_kw": obs,
                            },
                            expected={
                                "hvac_exp": exp_hvac,
                                "lighting_exp": exp_lighting,
                                "other_exp": exp_other,
                                "power_exp": exp,
                            },
                        ),
                    })
        anomalies.sort(key=lambda a: a["deviation_pct"], reverse=True)
        return anomalies[:40]


def peak_demand_warning(df: pd.DataFrame, threshold_kw: float | None = None) -> list[dict]:
    """Flag hours where power exceeds 95th percentile (peak demand warning)."""
    if df.empty:
        return []
    p95 = threshold_kw or float(np.percentile(df["power_kw"], 95))
    recent = df[df["timestamp"] > df["timestamp"].max() - timedelta(days=2)]
    warnings = []
    for _, row in recent.iterrows():
        if row["power_kw"] >= p95:
            warnings.append({
                "timestamp": row["timestamp"].isoformat(),
                "power_kw": round(float(row["power_kw"]), 2),
                "threshold_kw": round(p95, 2),
                "type": "PEAK_DEMAND_WARNING",
                "message": "Power approaching/d exceeding the 95th percentile threshold.",
            })
    return warnings


def simple_stats_anomalies(df: pd.DataFrame, zone_names: dict[int, str]) -> list[dict]:
    """Fallback 3-sigma detector on building-level power for baseline comparison."""
    if df.empty or len(df) < 50:
        return []
    recent = df.tail(72)
    mean = float(recent["power_kw"].mean())
    std = float(recent["power_kw"].std())
    out = []
    for _, row in recent.iterrows():
        if std > 0 and abs(row["power_kw"] - mean) > 2.5 * std:
            out.append({
                "zone_id": None,
                "zone_name": "Whole Building",
                "timestamp": row["timestamp"].isoformat(),
                "observed_power_kw": round(float(row["power_kw"]), 2),
                "expected_power_kw": round(mean, 2),
                "deviation_pct": round((row["power_kw"] - mean) / mean * 100, 1),
                "severity": "HIGH" if (row["power_kw"] - mean) > 3 * std else "MEDIUM",
                "isolation_score": 0.0,
                "possible_reason": "Building-level consumption far outside expected band.",
            })
    return out[:20]