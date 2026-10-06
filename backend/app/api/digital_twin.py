from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.ai.optimization import optimize_twin
from app.database.connection import get_db
from app.digital_twin.simulation import (
    apply_controls,
    simulate_building_day,
    simulate_zone_day,
)
from app.digital_twin.twin_service import build_twin_from_db, twin_snapshot
from app.schemas import BuildingSimulateRequest, SimulateRequest

router = APIRouter(prefix="/digital-twin")


@router.get("")
def get_twin(db: Session = Depends(get_db)):
    try:
        return twin_snapshot(db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Digital twin failure: {exc}")


@router.post("/simulate/zone")
def simulate_zone(req: SimulateRequest, db: Session = Depends(get_db)):
    try:
        twin, zone_energy_map = build_twin_from_db(db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    zone = twin.zone(name=req.zone)
    if not zone:
        raise HTTPException(status_code=404,
                            detail=f"Zone '{req.zone}' not found in digital twin.")
    try:
        archive = zone_energy_map.get(zone.name)
        result = simulate_zone_day(zone, req.hvac, req.lighting, req.occupancy,
                                   baseline_zone=zone, energy_from_archive=archive)
        # reflect new control state on the twin
        apply_controls(twin, zone.id, req.hvac, req.lighting, req.occupancy)
        # run one-hour state update so snapshot reflects control changes
        result["zone_state"] = {
            "id": zone.id,
            "name": zone.name,
            "occupancy_pct": round(zone.occupancy_pct, 1),
            "hvac_intensity_pct": round(zone.hvac_intensity_pct, 1),
            "lighting_pct": round(zone.lighting_pct, 1),
            "status": zone.status,
            "power_kw": zone.power_kw,
            "simulated_kw_after": result["simulated"]["energy_kwh"] / 24.0,
        }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/simulate")
def simulate_zone_compat(req: SimulateRequest, db: Session = Depends(get_db)):
    """Canonical single-zone simulation endpoint used by external clients."""
    return simulate_zone(req, db)


@router.post("/simulate/building")
def simulate_building(req: BuildingSimulateRequest, db: Session = Depends(get_db)):
    try:
        twin, zone_energy_map = build_twin_from_db(db)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    overrides = {name: ctrl.model_dump() for name, ctrl in req.zones.items()}
    for name in overrides:
        if not twin.zone(name=name):
            raise HTTPException(status_code=404, detail=f"Zone '{name}' not found in twin.")
    try:
        return simulate_building_day(twin, overrides, zone_energy_map)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/optimize")
def optimize(db: Session = Depends(get_db)):
    try:
        twin, zone_energy_map = build_twin_from_db(db)
        return optimize_twin(twin, zone_energy_map)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Optimization failure: {exc}")
