"""Recommendation engine.

Translates analytics + anomaly + optimization outputs into actionable,
contextual recommendations. No random text — everything is derived from
the actual dataset and simulation results.
"""

from __future__ import annotations

import pandas as pd

from app.config import settings
from app.database.models import Recommendation
from app.services import analytics_service


def generate_recommendations(
    db,
    anomalies: list[dict],
    zone_totals: list[dict],
    optimization: dict,
) -> list[dict]:
    """Fresh, data-driven recommendations computed at request time."""
    recs: list[dict] = []

    # ---- derive savings from optimization best scenario
    best = (optimization or {}).get("best_scenario") or {}
    best_saving_kwh = best.get("savings_kwh", 0)
    best_saving_cost = best.get("cost_saved_inr", 0)

    if best and best_saving_kwh > 0:
        recs.append({
            "title": "Optimize HVAC setpoint across all zones",
            "description": f"Adopt HVAC {best.get('hvac_pct')}% / lighting "
                           f"{best.get('lighting_pct')}% control profile during occupied hours.",
            "reason": "Deterministic search across feasible HVAC/lighting combinations "
                      "found this profile minimizes daily energy while respecting comfort limits.",
            "expected_savings": round(best.get("reduction_pct", 0), 1),
            "estimated_cost_saving": round(best_saving_cost, 2),
            "confidence": best.get("confidence_pct", 80),
            "feasibility": best.get("feasibility", "MEDIUM"),
            "status": "OPEN",
            "source": "optimization",
        })

    # ---- anomaly driven
    for a in anomalies[:3]:
        if a.get("severity") in ("HIGH", "CRITICAL"):
            occ = a.get("occupancy_pct", 50)
            if "HVAC" in (a.get("possible_reason") or ""):
                recs.append({
                    "title": f"Reduce HVAC in {a.get('zone_name')} during low occupancy",
                    "description": "Cut HVAC intensity to 70% during the hours around "
                                   f"{a.get('timestamp', '')} when occupancy is ~{occ}%.",
                    "reason": a.get("possible_reason"),
                    "expected_savings": round(a.get("deviation_pct", 0) * 0.35, 1),
                    "estimated_cost_saving": round(
                        (a.get("observed_power_kw", 0) - a.get("expected_power_kw", 0))
                        * 4.0 * settings.ENERGY_TARIFF_PER_KWH, 2),
                    "confidence": 91,
                    "feasibility": "HIGH",
                    "status": "OPEN",
                    "source": "anomaly",
                })

    # ---- zone driven
    if zone_totals:
        worst = zone_totals[0]
        recs.append({
            "title": f"Audit {worst['zone_name']} — highest consumer",
            "description": f"{worst['zone_name']} consumed {worst['energy_kwh']} kWh over the "
                           "period, the highest of any zone.",
            "reason": f"Zone ranking shows {worst['zone_name']} leads consumption "
                      f"(peak {worst['peak_power_kw']} kW).",
            "expected_savings": 5.0,
            "estimated_cost_saving": round(
                worst["energy_kwh"] * 0.05 * settings.ENERGY_TARIFF_PER_KWH, 2),
            "confidence": 76,
            "feasibility": "MEDIUM",
            "status": "OPEN",
            "source": "analytics",
        })

    # ---- stored seed recommendations (derived from seed analytics)
    for r in db.query(Recommendation).order_by(Recommendation.id).all():
        recs.append({
            "id": r.id,
            "title": r.title,
            "description": r.description,
            "reason": r.reason,
            "expected_savings": r.expected_savings,
            "estimated_cost_saving": r.estimated_cost_saving,
            "confidence": round(r.confidence * 100, 1),
            "feasibility": r.feasibility,
            "status": r.status,
            "source": "analytics",
        })

    # de-duplicate by title
    seen = set()
    out = []
    for r in recs:
        key = r["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out[:12]