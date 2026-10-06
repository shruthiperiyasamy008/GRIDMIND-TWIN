"""Simulated energy data generator.

Generates realistic hourly energy readings for the WattWise Demo Campus.

IMPORTANT: This data is SIMULATED and is labelled as such everywhere in the
application. It is designed to mimic realistic building energy behaviour so
that the AI / ML / digital-twin pipeline can be exercised end-to-end. It is
NOT real smart-meter data.
"""

import random
from datetime import datetime, timedelta

import numpy as np

random.seed(42)
np.random.seed(42)


def _daily_profile(hour: int, day_of_week: int) -> float:
    """Base activity multiplier from an hour of the day and weekday."""
    is_weekend = day_of_week >= 5
    if is_weekend:
        base = [0.10, 0.08, 0.07, 0.06, 0.06, 0.07, 0.10, 0.18, 0.28, 0.34,
                0.36, 0.40, 0.42, 0.44, 0.42, 0.40, 0.36, 0.30, 0.24, 0.20,
                0.16, 0.13, 0.11, 0.10]
    else:
        base = [0.08, 0.07, 0.06, 0.06, 0.06, 0.10, 0.30, 0.58, 0.78, 0.88,
                0.92, 0.95, 0.96, 0.94, 0.90, 0.82, 0.72, 0.60, 0.48, 0.40,
                0.32, 0.24, 0.15, 0.10]
    return base[hour]


def generate_zone_hourly(
    zone_base_kw: float,
    start: datetime,
    hours: int,
    zone_index: int,
    hvac_weight: float,
) -> list[dict]:
    """Generate one row per hour for a single zone."""
    rows = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        dow = ts.weekday()
        hour = ts.hour
        activity = _daily_profile(hour, dow)

        seasonal = 1.0 + 0.12 * np.sin(2 * np.pi * (ts.timetuple().tm_yday - 200) / 365.0)
        temp = 26.0 if not (hour >= 6 and hour < 20 and dow < 5) else 29.0 + 6.0 * np.sin(
            np.pi * (hour - 6) / 14.0
        )
        temp = float(np.clip(temp + random.uniform(-1.5, 1.5), 14, 38))

        occupancy = float(np.clip(activity * random.uniform(0.85, 1.1), 0, 1) * 100)

        hvac_load = float(zone_base_kw * hvac_weight * activity)
        lighting_load = float(zone_base_kw * 0.18 * (activity + 0.1) * random.uniform(0.9, 1.1))
        pump_load = float(zone_base_kw * 0.08 * (0.4 + 0.6 * activity))
        other_load = float(zone_base_kw * 0.14 * (0.3 + 0.7 * activity) * random.uniform(0.9, 1.1))
        power_kw = float(hvac_load + lighting_load + pump_load + other_load)

        rows.append({
            "timestamp": ts,
            "zone_index_offset": zone_index,
            "power_kw": round(power_kw, 2),
            "energy_kwh": round(power_kw, 2),
            "temperature": round(temp, 1),
            "occupancy": round(occupancy, 1),
            "hvac_load": round(hvac_load, 2),
            "lighting_load": round(lighting_load, 2),
            "pump_load": round(pump_load, 2),
            "other_load": round(other_load, 2),
        })
    return rows


ZONE_CONFIG = [
    ("Block A Classroom 1", 82.0, 0.62, "classroom"),
    ("Block A Classroom 2", 78.0, 0.62, "classroom"),
    ("Block B Classroom 3", 76.0, 0.62, "classroom"),
    ("Block B Classroom 4", 74.0, 0.62, "classroom"),
    ("Computer Lab", 88.0, 0.50, "computer-lab"),
    ("Administration", 55.0, 0.55, "office"),
    ("Library", 48.0, 0.52, "library"),
    ("Pump Room", 30.0, 0.10, "utility"),
]

ANOMALY_DEFS = [
    # (zone_index, day_offset, local_hour, boost_kw, reason)
    (0, 9, 14, 320, "HVAC running at full capacity during low occupancy"),
    (0, 9, 15, 300, "HVAC running at full capacity during low occupancy"),
    (4, 12, 11, 240, "Abnormal compute / cooling load outside schedule"),
    (5, 14, 16, 200, "Unusually high lighting load during low activity"),
    (1, 16, 10, 200, "Abnormal equipment runtime outside schedule"),
]


def generate_all_hourly(start: datetime, days: int = 30) -> list[dict]:
    hours = days * 24
    all_rows = []
    for zi, (name, base, hvac_w, ztype) in enumerate(ZONE_CONFIG):
        rows = generate_zone_hourly(base, start, hours, zi, hvac_w)
        # Apply anomaly injections anchored to local wall-clock time
        def anomaly_target(day_offset: int, local_hour: int) -> int:
            target = (start.date() + timedelta(days=day_offset, hours=local_hour))
            return int((datetime.combine(target.date(), target.time()) - start).total_seconds() // 3600)

        for (azi, day, hour, boost, _reason) in ANOMALY_DEFS:
            if azi != zi:
                continue
            idx = anomaly_target(day, hour)
            if 0 <= idx < hours:
                rows[idx]["hvac_load"] = round(rows[idx]["hvac_load"] + boost, 2)
                rows[idx]["power_kw"] = round(rows[idx]["power_kw"] + boost, 2)
                rows[idx]["anomaly_reason"] = _reason
                if idx + 1 < hours:
                    rows[idx + 1]["hvac_load"] = round(rows[idx + 1]["hvac_load"] + boost * 0.6, 2)
                    rows[idx + 1]["power_kw"] = round(rows[idx + 1]["power_kw"] + boost * 0.6, 2)
                    rows[idx + 1]["anomaly_reason"] = _reason
        all_rows.extend(rows)
    return all_rows


def aggregate_building(rows: list[dict], start: datetime, days: int) -> list[dict]:
    """Aggregate zone rows into building-level hourly rows."""
    hours = days * 24
    building_rows = []
    for i in range(hours):
        ts = start + timedelta(hours=i)
        zone_rows = [r for r in rows if r["timestamp"] == ts]
        if not zone_rows:
            continue
        power = sum(r["power_kw"] for r in zone_rows)
        energy = sum(r["energy_kwh"] for r in zone_rows)
        temp = float(np.mean([r["temperature"] for r in zone_rows]))
        occ = float(np.mean([r["occupancy"] for r in zone_rows]))
        building_rows.append({
            "timestamp": ts,
            "power_kw": round(power, 2),
            "energy_kwh": round(energy, 2),
            "temperature": round(temp, 1),
            "occupancy": round(occ, 1),
            "hvac_load": round(sum(r["hvac_load"] for r in zone_rows), 2),
            "lighting_load": round(sum(r["lighting_load"] for r in zone_rows), 2),
            "pump_load": round(sum(r["pump_load"] for r in zone_rows), 2),
            "other_load": round(sum(r["other_load"] for r in zone_rows), 2),
        })
    return building_rows


def to_csv(rows: list[dict], path: str):
    import pandas as pd

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


if __name__ == "__main__":
    import os

    start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=30)
    demo_out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "energy_data.csv"
    )
    rows = generate_all_hourly(start, days=30)
    to_csv(rows, demo_out)
    print(f"Wrote {len(rows)} rows -> {os.path.abspath(demo_out)}")