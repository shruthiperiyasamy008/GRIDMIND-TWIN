"""Structured digital-twin model of the building.

A Building is a tree of zones; each zone holds equipment plus state
(occupancy, temperature, HVAC intensity, lighting intensity). The
simulation engine derives loads from these physics-based inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EquipmentState:
    name: str
    equipment_type: str
    status: str = "ON"
    power_rating_kw: float = 0.0


@dataclass
class ZoneState:
    id: int
    name: str
    zone_type: str
    area_m2: float
    x: float
    y: float
    w: float
    h: float
    equipment: list[EquipmentState] = field(default_factory=list)

    # Control state (drives simulation)
    occupancy_pct: float = 50.0
    hvac_intensity_pct: float = 100.0
    lighting_pct: float = 100.0
    temperature_c: float = 26.0
    operating_schedule: str = "07:00-21:00"
    is_running: bool = True

    # Calibration: scale factor aligning physics-model output with observed
    # (simulated) historical magnitudes for this zone.
    calib_factor: float = 1.0

    # Result state (filled by simulation)
    power_kw: float = 0.0
    energy_today_kwh: float = 0.0
    status: str = "NORMAL"

    @property
    def status_class(self) -> str:
        return self.status


@dataclass
class BuildingTwin:
    id: int
    name: str
    zones: list[ZoneState] = field(default_factory=list)

    def zone(self, name: str | None = None, zone_id: int | None = None) -> ZoneState | None:
        for z in self.zones:
            if name is not None and z.name.lower() == name.lower():
                return z
            if zone_id is not None and z.id == zone_id:
                return z
        return None

    def total_power_kw(self) -> float:
        return sum(z.power_kw for z in self.zones)

    def total_energy_today_kwh(self) -> float:
        return sum(z.energy_today_kwh for z in self.zones)


# ---------------------------------------------------------------- defaults


DEFAULTS: dict[str, dict] = {
    "computer-lab": {
        "base_load_kw": 4.5, "hvac_frac": 0.52, "lighting_w_m2": 6.5,
        "pump_frac": 0.03, "other_frac": 0.10, "hvac_cop": 3.2,
        "computer_load_frac": 0.30,
    },
    "classroom": {
        "base_load_kw": 2.0, "hvac_frac": 0.60, "lighting_w_m2": 7.0,
        "pump_frac": 0.03, "other_frac": 0.06, "hvac_cop": 3.0,
        "computer_load_frac": 0.06,
    },
    "office": {
        "base_load_kw": 2.2, "hvac_frac": 0.56, "lighting_w_m2": 6.0,
        "pump_frac": 0.03, "other_frac": 0.12, "hvac_cop": 3.1,
        "computer_load_frac": 0.14,
    },
    "library": {
        "base_load_kw": 2.0, "hvac_frac": 0.54, "lighting_w_m2": 5.5,
        "pump_frac": 0.03, "other_frac": 0.10, "hvac_cop": 3.0,
        "computer_load_frac": 0.08,
    },
    "utility": {
        "base_load_kw": 1.5, "hvac_frac": 0.10, "lighting_w_m2": 3.0,
        "pump_frac": 0.45, "other_frac": 0.05, "hvac_cop": 2.6,
        "computer_load_frac": 0.02,
    },
}


def _hour_activity(hour: int, weekday: bool) -> float:
    if not weekday:
        profile = [0.08, 0.07, 0.06, 0.06, 0.06, 0.07, 0.10, 0.18, 0.28, 0.34,
                   0.36, 0.40, 0.42, 0.44, 0.42, 0.40, 0.36, 0.30, 0.24, 0.20,
                   0.16, 0.13, 0.11, 0.10]
    else:
        profile = [0.05, 0.05, 0.05, 0.05, 0.06, 0.10, 0.32, 0.60, 0.80, 0.90,
                   0.94, 0.96, 0.97, 0.95, 0.90, 0.82, 0.72, 0.60, 0.48, 0.40,
                   0.32, 0.24, 0.15, 0.08]
    return profile[int(hour) % 24]


class TwinFactory:
    """Builds the BuildingTwin from DB zones/equipment."""

    @staticmethod
    def from_db(building, zones, equipment_by_zone) -> BuildingTwin:
        layout = TwinFactory._layout(len(zones))
        twin = BuildingTwin(id=building.id, name=building.name)
        for i, z in enumerate(zones):
            specs = DEFAULTS.get(z.zone_type, DEFAULTS["classroom"])
            lx, ly, lw, lh = layout[i % len(layout)]
            eqs = [EquipmentState(
                name=e.name,
                equipment_type=e.equipment_type,
                status=e.status,
                power_rating_kw=e.power_rating,
            ) for e in equipment_by_zone.get(z.id, [])]
            twin.zones.append(ZoneState(
                id=z.id, name=z.name, zone_type=z.zone_type, area_m2=z.area,
                x=lx, y=ly, w=lw, h=lh, equipment=eqs,
            ))
        TwinFactory._initialise_states(twin)
        return twin

    @staticmethod
    def _layout(n: int) -> list[tuple[float, float, float, float]]:
        # coarse blueprint geometry (updated by frontend layout)
        grid = [(1, 1, 3.4, 2.0), (4.7, 1, 3.4, 2.0), (1, 3.4, 3.4, 2.0),
                (4.7, 3.4, 3.4, 2.0), (8.4, 1, 3.2, 2.0), (8.4, 3.4, 3.2, 2.0),
                (1, 6.0, 5.6, 1.9), (6.9, 6.0, 4.8, 1.9)]
        return grid[: max(n, len(grid)):]

    @staticmethod
    def _initialise_states(twin: BuildingTwin):
        import numpy as np

        rng = np.random.default_rng(7)
        for z in twin.zones:
            # Stable baseline controls make the primary demo scenario repeatable.
            is_block_a = z.name.startswith("Block A")
            z.occupancy_pct = 35.0 if is_block_a else float(rng.uniform(30, 60))
            z.hvac_intensity_pct = 100.0
            z.lighting_pct = 80.0 if is_block_a else 100.0
            z.temperature_c = float(rng.uniform(24.5, 28.5))


def step_simulation_zone(zone: ZoneState, ambient_temp: float, hour: int,
                         weekday: bool) -> ZoneState:
    """Physics-inspired computation of instantaneous load for a zone."""
    specs = DEFAULTS.get(zone.zone_type, DEFAULTS["classroom"])

    activity = _hour_activity(hour, weekday)
    # occupancy driven both by schedule + the user-set occupancy slider
    occ = np.clip(zone.occupancy_pct / 100.0, 0, 1)
    occupant_effect = 0.5 + 0.5 * occ
    schedule_active = zone.is_running
    intensity = np.clip(zone.hvac_intensity_pct / 100.0, 0, 1)
    light = np.clip(zone.lighting_pct / 100.0, 0, 1)

    # temperature delta against comfort setpoint 24C
    delta_t = max(0.0, ambient_temp - 24.0)
    heat_from_occupants = 0.45 * occ * zone.area_m2 * 0.09

    # HVAC: cooling load proportional to (heat gain) x (demand) with COP
    cooling_need = (delta_t * 0.55 + heat_from_occupants) * activity * occupant_effect
    hvac_power = (cooling_need * zone.area_m2 / 60.0) / specs["hvac_cop"] * intensity
    hvac_power = hvac_power if schedule_active and intensity > 0.01 else 0.0

    # Lighting: area x intensity (0.4 = daylight factor)
    lighting_power = (specs["lighting_w_m2"] * zone.area_m2 / 1000.0) * light
    lighting_power = lighting_power if schedule_active else lighting_power * 0.05

    # Computers: footprint within zone config
    base = specs["base_load_kw"]
    computer_power = specs["computer_load_frac"] * base * activity * occupant_effect
    # Pumps: constant-ish duty scaled by schedule
    pump_power = specs["pump_frac"] * base * (0.5 if schedule_active else 0.2)
    other_power = specs["other_frac"] * base * (0.4 + 0.6 * activity)

    total = hvac_power + lighting_power + computer_power + pump_power + other_power
    total = total * max(zone.calib_factor, 1e-6)

    zone.power_kw = round(float(total), 2)
    zone.energy_today_kwh = max(0.0, float(round(zone.energy_today_kwh + total, 2)))
    zone.status = "ANOMALY" if total > 60.0 else ("HIGH" if total > 40.0 else "NORMAL")
    return zone


def simulate_hour(twin: BuildingTwin, ambient_temp: float, hour: int,
                  weekday: bool) -> BuildingTwin:
    for z in twin.zones:
        step_simulation_zone(z, ambient_temp, hour, weekday)
    return twin
