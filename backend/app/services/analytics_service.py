"""Analytics service: loads data from DB into pandas frames and exposes
aggregations needed by the API and the AI modules.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from app.database.models import Building, EnergyReading, Zone
from app.services import energy_service
from app.services.data_provider import SimulatedDataProvider


def readings_frame(
    db: Session,
    start: datetime | None = None,
    end: datetime | None = None,
    building_id: int | None = None,
    zone_id: int | None = None,
) -> pd.DataFrame:
    q = db.query(EnergyReading)
    if start:
        q = q.filter(EnergyReading.timestamp >= start)
    if end:
        q = q.filter(EnergyReading.timestamp <= end)
    if building_id:
        q = q.filter(EnergyReading.building_id == building_id)
    if zone_id:
        q = q.filter(EnergyReading.zone_id == zone_id)
    rows = q.order_by(EnergyReading.timestamp).all()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "timestamp": r.timestamp,
        "building_id": r.building_id,
        "zone_id": r.zone_id,
        "power_kw": r.power_kw,
        "energy_kwh": r.energy_kwh,
        "temperature": r.temperature,
        "occupancy": r.occupancy,
        "hvac_load": r.hvac_load,
        "lighting_load": r.lighting_load,
        "pump_load": r.pump_load,
        "other_load": r.other_load,
    } for r in rows])
    return energy_service.preprocess(df)


def via_provider(db: Session, building_id: int | None = None, zone_id: int | None = None) -> SimulatedDataProvider:
    """Return a DataProvider backed by the DB readings."""
    start = datetime.now() - timedelta(days=60)
    df = readings_frame(db, start=start, building_id=building_id, zone_id=zone_id)
    rows = df.to_dict(orient="records")
    return SimulatedDataProvider(rows)


def building_level_frame(db: Session, start=None, end=None, building_id: int | None = None) -> pd.DataFrame:
    df = readings_frame(db, start=start, end=end, building_id=building_id)
    if df.empty:
        return df
    return energy_service.aggregate_building_hourly(df)


def zone_map(db: Session) -> dict[int, Zone]:
    return {z.id: z for z in db.query(Zone).all()}


def building_map(db: Session) -> dict[int, Building]:
    return {b.id: b for b in db.query(Building).all()}


def zone_totals(db: Session, start=None, end=None) -> list[dict]:
    """Per-zone aggregated energy, joined with names."""
    df = readings_frame(db, start=start, end=end)
    if df.empty:
        return []
    zones = zone_map(db)
    rows = df.groupby("zone_id").agg(
        energy_kwh=("energy_kwh", "sum"),
        mean_power=("power_kw", "mean"),
        peak_power=("power_kw", "max"),
    ).reset_index()
    out = []
    for r in rows.to_dict(orient="records"):
        z = zones.get(r["zone_id"])
        out.append({
            "zone_id": r["zone_id"],
            "zone_name": z.name if z else f"Zone {r['zone_id']}",
            "energy_kwh": round(float(r["energy_kwh"]), 2),
            "mean_power_kw": round(float(r["mean_power"]), 2),
            "peak_power_kw": round(float(r["peak_power"]), 2),
        })
    return sorted(out, key=lambda x: x["energy_kwh"], reverse=True)