"""Data source abstraction.

Future real-world providers (smart meter, IoT, BMS) can implement the same
interface so the analytics pipeline stays provider-agnostic. Only the
SimulatedDataProvider ships with the MVP.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


@dataclass
class MeterSample:
    timestamp: datetime
    building_id: int
    zone_id: int
    power_kw: float
    energy_kwh: float
    temperature: float
    occupancy: float
    hvac_load: float
    lighting_load: float
    pump_load: float
    other_load: float


class EnergyDataProvider(ABC):
    """Interface every energy data source must implement."""

    name: str = "abstract"

    @abstractmethod
    def fetch_historical(
        self,
        start: datetime,
        end: datetime,
        building_id: int | None = None,
        zone_id: int | None = None,
    ) -> list[MeterSample]:
        ...

    @abstractmethod
    def provider_label(self) -> str:
        ...


class SimulatedDataProvider(EnergyDataProvider):
    """Reads DEMO / SIMULATED energy data. Never label this as real meter data."""

    name = "SimulatedDataProvider"

    def __init__(self, rows: list[dict]):
        self._rows = rows

    def provider_label(self) -> str:
        return "SIMULATED"

    def fetch_historical(
        self,
        start: datetime,
        end: datetime,
        building_id: int | None = None,
        zone_id: int | None = None,
    ) -> list[MeterSample]:
        out = []
        for r in self._rows:
            if start <= r["timestamp"] <= end:
                if building_id is not None and r.get("building_id") != building_id:
                    continue
                if zone_id is not None and r.get("zone_id") != zone_id:
                    continue
                out.append(
                    MeterSample(
                        timestamp=r["timestamp"],
                        building_id=r.get("building_id", 1),
                        zone_id=r.get("zone_id", 1),
                        power_kw=r["power_kw"],
                        energy_kwh=r["energy_kwh"],
                        temperature=r.get("temperature", 25.0),
                        occupancy=r.get("occupancy", 0.0),
                        hvac_load=r.get("hvac_load", 0.0),
                        lighting_load=r.get("lighting_load", 0.0),
                        pump_load=r.get("pump_load", 0.0),
                        other_load=r.get("other_load", 0.0),
                    )
                )
        return out

    # Placeholder for clearly-labelled future providers (never faked as live).
    # class SmartMeterProvider(EnergyDataProvider): ...        # future
    # class IoTProvider(EnergyDataProvider): ...               # future
    # class BuildingManagementSystemProvider(EnergyDataProvider): ...  # future


def build_provider_from_rows(rows: list[dict]) -> SimulatedDataProvider:
    return SimulatedDataProvider(rows)