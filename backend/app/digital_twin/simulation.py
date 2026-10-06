"""Digital-twin simulation engine.

Given a control action (HVAC %, lighting %, occupancy %), runs the physics
inspired zone model across a simulated 24-hour day and returns the resulting
energy, cost and savings versus the baseline state. All outputs are labelled
ESTIMATED / SIMULATED SCENARIO.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime

import numpy as np

from app.config import settings
from app.digital_twin.building_model import (
    BuildingTwin,
    ZoneState,
    simulate_hour,
)


def _temperature_at_hour(hour: int) -> float:
    """Typical daily ambient temperature curve for the campus."""
    if 6 <= hour <= 20:
        return 29.0 + 6.0 * np.sin(np.pi * (hour - 6) / 14.0)
    return 26.0


def _simulate_day(zone: ZoneState, hvac_pct: float, lighting_pct: float,
                  occupancy_pct: float) -> list[dict]:
    """Run the zone physics model over 24 hours; return hourly rows."""
    work = ZoneState(
        id=zone.id, name=zone.name, zone_type=zone.zone_type,
        area_m2=zone.area_m2, x=zone.x, y=zone.y, w=zone.w, h=zone.h,
        equipment=[deepcopy(e) for e in zone.equipment],
        occupancy_pct=float(occupancy_pct),
        hvac_intensity_pct=float(hvac_pct),
        lighting_pct=float(lighting_pct),
        temperature_c=zone.temperature_c,
        operating_schedule=zone.operating_schedule,
        is_running=zone.is_running,
        calib_factor=zone.calib_factor,
    )
    hours: list[dict] = []
    for hour in range(24):
        ambient = _temperature_at_hour(hour)
        simulate_hour(
            BuildingTwin(id=0, name="sim", zones=[work]),
            ambient, hour, datetime.now().weekday() < 5,
        )
        hours.append({
            "hour": hour,
            "power_kw": round(work.power_kw, 2),
            "energy_kwh": round(work.power_kw, 2),
            "temperature_c": round(ambient, 1),
        })
    return hours


def simulate_zone_day(
    zone: ZoneState,
    hvac_pct: float,
    lighting_pct: float,
    occupancy_pct: float,
    baseline_zone: ZoneState | None = None,
    energy_from_archive: float | None = None,
) -> dict:
    """Simulate a full day for one zone under the given control inputs."""
    hvac_pct = float(hvac_pct)
    lighting_pct = float(lighting_pct)
    occupancy_pct = float(occupancy_pct)
    if not (0 <= hvac_pct <= 120):
        raise ValueError("HVAC percentage must be between 0 and 120.")
    if not (0 <= lighting_pct <= 100):
        raise ValueError("Lighting percentage must be between 0 and 100.")
    if not (0 <= occupancy_pct <= 100):
        raise ValueError("Occupancy percentage must be between 0 and 100.")

    sim_hours = _simulate_day(zone, hvac_pct, lighting_pct, occupancy_pct)
    base_hours = _simulate_day(
        baseline_zone or zone,
        hvac_pct=(baseline_zone.hvac_intensity_pct if baseline_zone else 100.0),
        lighting_pct=(baseline_zone.lighting_pct if baseline_zone else 100.0),
        occupancy_pct=(baseline_zone.occupancy_pct if baseline_zone else 50.0),
    )

    total_energy = round(sum(s["energy_kwh"] for s in sim_hours), 2)
    base_energy = energy_from_archive or round(sum(b["energy_kwh"] for b in base_hours), 2)

    savings = round(base_energy - total_energy, 2)
    cost_savings = round(max(0.0, savings) * settings.ENERGY_TARIFF_PER_KWH, 2)
    reduction_pct = round(savings / base_energy * 100.0, 1) if base_energy > 0 else 0.0

    return {
        "zone": zone.name,
        "inputs": {
            "hvac_pct": hvac_pct,
            "lighting_pct": lighting_pct,
            "occupancy_pct": occupancy_pct,
            "schedule": zone.operating_schedule,
        },
        "baseline": {
            "energy_kwh": round(base_energy, 2),
            "cost_inr": round(base_energy * settings.ENERGY_TARIFF_PER_KWH, 2),
        },
        "simulated": {
            "energy_kwh": total_energy,
            "cost_inr": round(total_energy * settings.ENERGY_TARIFF_PER_KWH, 2),
            "energy_saved_kwh": savings,
            "estimated_cost_saved_inr": cost_savings,
            "reduction_pct": reduction_pct,
        },
        "hourly": sim_hours,
        "peak_kw_simulated": round(max(s["power_kw"] for s in sim_hours), 2),
        "labels": {
            "energy": "ESTIMATED",
            "scenario": "SIMULATED SCENARIO",
            "cost": f"ESTIMATED at ₹{settings.ENERGY_TARIFF_PER_KWH}/kWh",
        },
    }


def simulate_building_day(
    twin: BuildingTwin,
    zone_overrides: dict[str, dict],
    zone_energy_map: dict[str, float] | None = None,
) -> dict:
    """Simulate whole-building day. zone_overrides maps zone name -> controls."""
    zone_energy_map = zone_energy_map or {}
    results: list[dict] = []
    grand_base = 0.0
    grand_sim = 0.0
    peak_sim = 0.0
    peak_base = 0.0
    for z in twin.zones:
        over = zone_overrides.get(z.name, {})
        hvac = float(over.get("hvac", z.hvac_intensity_pct))
        lighting = float(over.get("lighting", z.lighting_pct))
        occ = float(over.get("occupancy", z.occupancy_pct))
        archive = zone_energy_map.get(z.name)
        res = simulate_zone_day(z, hvac, lighting, occ,
                                baseline_zone=z, energy_from_archive=archive)
        results.append(res)
        grand_base += res["baseline"]["energy_kwh"]
        grand_sim += res["simulated"]["energy_kwh"]
        peak_sim = max(peak_sim, res["peak_kw_simulated"])
        peak_base = max(peak_base, res["baseline"]["energy_kwh"] / 24.0)

    savings = round(grand_base - grand_sim, 2)
    cost_savings = round(max(0.0, savings) * settings.ENERGY_TARIFF_PER_KWH, 2)
    # impact in peak impact terms
    peak_change = round(peak_base - peak_sim, 2)
    return {
        "building": twin.name,
        "baseline": {
            "energy_kwh": round(grand_base, 2),
            "cost_inr": round(grand_base * settings.ENERGY_TARIFF_PER_KWH, 2),
            "peak_kw": round(peak_base, 2),
        },
        "simulated": {
            "energy_kwh": round(grand_sim, 2),
            "cost_inr": round(grand_sim * settings.ENERGY_TARIFF_PER_KWH, 2),
            "peak_kw": round(peak_sim, 2),
            "peak_change_kw": peak_change,
            "energy_saved_kwh": savings,
            "estimated_cost_saved_inr": cost_savings,
            "reduction_pct": round(savings / grand_base * 100.0, 1) if grand_base > 0 else 0.0,
        },
        "zones": results,
        "labels": {"scenario": "SIMULATED SCENARIO", "savings": "ESTIMATED"},
    }


def apply_controls(twin: BuildingTwin, zone_id: int, hvac: float,
                   lighting: float, occupancy: float) -> BuildingTwin:
    """Mutate the twin's control inputs without re-simulating."""
    z = twin.zone(zone_id=zone_id)
    if not z:
        raise ValueError(f"Zone id {zone_id} not found in twin.")
    z.hvac_intensity_pct = float(hvac)
    z.lighting_pct = float(lighting)
    z.occupancy_pct = float(occupancy)
    return twin