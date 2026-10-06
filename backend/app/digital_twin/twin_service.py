"""Service that links the DB state to a BuildingTwin instance."""

from __future__ import annotations

from datetime import timedelta

from sqlalchemy.orm import Session

from app.database.models import Building, Equipment, Zone
from app.digital_twin.building_model import BuildingTwin, TwinFactory
from app.digital_twin.simulation import (
    _simulate_day,
    apply_controls,
    simulate_building_day,
    simulate_zone_day,
)
from app.services import analytics_service


def build_twin_from_db(db: Session) -> tuple[BuildingTwin, dict[str, float]]:
    building = db.query(Building).order_by(Building.id).first()
    if not building:
        raise ValueError("No buildings seeded. Please seed the database first.")
    zones = db.query(Zone).filter(Zone.building_id == building.id).order_by(Zone.id).all()
    eq_rows = db.query(Equipment).all()
    equipment_by_zone: dict[int, list[Equipment]] = {}
    for e in eq_rows:
        equipment_by_zone.setdefault(e.zone_id, []).append(e)

    twin = TwinFactory.from_db(building, zones, equipment_by_zone)

    # archive per-zone energy for consistent baselines
    zone_energy_map: dict[str, float] = {}
    df = analytics_service.readings_frame(db, start=None)
    if not df.empty:
        cutoff = df["timestamp"].max() - timedelta(days=3)
        recent = df[df["timestamp"] >= cutoff]
        g = recent.groupby("zone_id").agg(energy=("energy_kwh", "sum")).reset_index()
        zone_names = {z.id: z.name for z in zones}
        for _, r in g.iterrows():
            # store DAILY average (last 3 days summed / 3)
            zone_energy_map[zone_names.get(int(r["zone_id"]), str(r["zone_id"]))] = float(r["energy"]) / 3.0

    calibrate_twin_from_archive(twin, zone_energy_map)
    return twin, zone_energy_map


def calibrate_twin_from_archive(twin: BuildingTwin, zone_energy_map: dict[str, float]) -> None:
    """Calibrate the physics model to archive daily energy magnitudes so twin
    outputs are consistent with the (simulated) meter history. Relative changes
    from control inputs remain physics-driven."""
    import numpy as np

    for z in twin.zones:
        archive_daily = zone_energy_map.get(z.name)
        if not archive_daily or archive_daily <= 0:
            continue
        model_energy = sum(s["energy_kwh"] for s in _simulate_day(z, 100.0, 100.0, z.occupancy_pct))
        if model_energy <= 0:
            continue
        calib = archive_daily / model_energy
        z.calib_factor = float(np.clip(calib, 0.3, 40.0))


def twin_snapshot(db: Session) -> dict:
    from datetime import datetime

    from app.digital_twin.building_model import step_simulation_zone
    from app.digital_twin.simulation import _temperature_at_hour as amb_curve

    twin, zone_energy_map = build_twin_from_db(db)
    now = datetime.now()
    current_hour = now.hour
    # apply current state so blueprint shows live (simulated) zone state
    for z in twin.zones:
        step_simulation_zone(z, amb_curve(current_hour), current_hour, now.weekday() < 5)

    zone_rows = []
    for z in twin.zones:
        zone_rows.append({
            "id": z.id,
            "name": z.name,
            "zone_type": z.zone_type,
            "x": z.x, "y": z.y, "w": z.w, "h": z.h,
            "occupancy_pct": round(z.occupancy_pct, 1),
            "hvac_intensity_pct": round(z.hvac_intensity_pct, 1),
            "lighting_pct": round(z.lighting_pct, 1),
            "temperature_c": round(z.temperature_c, 1),
            "operating_schedule": z.operating_schedule,
            "is_running": z.is_running,
            "power_kw": round(z.power_kw, 2),
            "energy_today_kwh": round(z.energy_today_kwh, 2),
            "status": z.status,
            "equipment": [
                {"name": e.name, "type": e.equipment_type, "status": e.status,
                 "rating_kw": e.power_rating_kw}
                for e in z.equipment
            ],
            "labels": {"source": "SIMULATED"},
        })
    return {
        "building": {
            "id": twin.id,
            "name": twin.name,
            "source": "SIMULATED",
        },
        "zones": zone_rows,
        "labels": {"source": "SIMULATED", "type": "SIMULATED SCENARIO"},
    }