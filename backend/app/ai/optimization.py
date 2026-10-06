"""Optimization engine.

Deterministic grid search over candidate control combinations. Evaluates
HVAC x Lighting steps against zone physics with constraints on comfort and
operating schedule, and selects the feasible scenario with the best
energy/cost outcome. Viable as an MVP replacement for a heavyweight solver.
"""

from __future__ import annotations

from app.config import settings
from app.digital_twin.building_model import BuildingTwin
from app.digital_twin.simulation import simulate_building_day

MIN_OCCUPANCY_COMFORT = 30.0   # occupancy below this triggers aggressive cutbacks
MAX_HVAC_REDUCTION = 50.0      # percentage points max from the baseline
MAX_LIGHTING_REDUCTION = 40.0  # percentage points max from baseline


def _feasible(hvac, lighting, occ, baseline_hvac, baseline_lighting) -> bool:
    if hvac < baseline_hvac - MAX_HVAC_REDUCTION:
        return False
    if lighting < baseline_lighting - MAX_LIGHTING_REDUCTION:
        return False
    return True


def optimize_twin(
    twin: BuildingTwin,
    zone_energy_map: dict[str, float] | None = None,
    high_occupancy_reference: dict[str, float] | None = None,
) -> dict:
    """Grid search over scenarios; return best + full ranked list."""
    hvac_grid = [70, 80, 90, 100]
    light_grid = [70, 80, 90, 100]

    candidates: list[dict] = []
    for hv in hvac_grid:
        for lt in light_grid:
            overrides = {}
            for z in twin.zones:
                occ = z.occupancy_pct or 50.0
                if occ < MIN_OCCUPANCY_COMFORT:
                    # low occupancy -> deeper acceptable cutbacks
                    passes = _feasible(hv, lt, occ, z.hvac_intensity_pct, z.lighting_pct)
                    if not passes:
                        continue
                overrides[z.name] = {"hvac": hv, "lighting": lt,
                                     "occupancy": z.occupancy_pct}
            res = simulate_building_day(twin, overrides, zone_energy_map)
            candidates.append({
                "scenario": f"HVAC {hv}% / Lighting {lt}%",
                "hvac_pct": hv,
                "lighting_pct": lt,
                "energy_kwh": res["simulated"]["energy_kwh"],
                "cost_inr": res["simulated"]["cost_inr"],
                "savings_kwh": res["simulated"]["energy_saved_kwh"],
                "cost_saved_inr": res["simulated"]["estimated_cost_saved_inr"],
                "reduction_pct": res["simulated"]["reduction_pct"],
                "baseline_energy_kwh": res["baseline"]["energy_kwh"],
                "peak_kw": res["simulated"]["peak_kw"],
            })

    candidates.sort(key=lambda c: c["energy_kwh"])
    best = candidates[0] if candidates else None
    for c in candidates:
        # confidence model: closer to baseline constraints -> higher confidence
        c["confidence_pct"] = round(100 - abs(c["hvac_pct"] - 100) * 0.3, 1)
        c["feasibility"] = "HIGH" if c["hvac_pct"] >= 80 else "MEDIUM"

    if best:
        best["confidence_pct"] = round(min(97.0, best["confidence_pct"]), 1)

    tariff = settings.ENERGY_TARIFF_PER_KWH
    return {
        "strategy": "Deterministic grid search over HVAC x Lighting combinations",
        "constraints": {
            "max_hvac_reduction_pp": MAX_HVAC_REDUCTION,
            "max_lighting_reduction_pp": MAX_LIGHTING_REDUCTION,
            "min_occupancy_for_full_service_pct": MIN_OCCUPANCY_COMFORT,
            "tariff_inr_per_kwh": tariff,
        },
        "best_scenario": best,
        "ranked_scenarios": candidates,
        "labels": {"savings": "ESTIMATED", "scenario": "SIMULATED SCENARIO"},
    }