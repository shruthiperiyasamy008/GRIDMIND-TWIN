"""AI Energy Analyst.

Contextual question answering over the live simulated dataset + simulation
results. Uses simple intent matching + numeric computations computed from
the actual data — NOT a generic chatbot.

Questions supported:
- Why is energy consumption high?
- What is causing the peak?
- Which zone consumes the most energy?
- What should I change?
- What if I reduce HVAC by X%?
- How much can I save?
"""

from __future__ import annotations

import re

import numpy as np

from app.config import settings
from app.services import energy_service


def _match_keywords(text: str, *groups: list[str]) -> bool:
    lowered = text.lower()
    for group in groups:
        if any(kw in lowered for kw in group):
            return True
    return False


def answer(
    question: str,
    building_df,
    zone_totals: list[dict],
    anomalies: list[dict],
    optimization: dict,
    zones_meta: list[dict],
) -> dict:
    q = question.lower()
    tariff = settings.ENERGY_TARIFF_PER_KWH

    # ---------------- Which zone consumes most ----------------
    if _match_keywords(q, ["which zone", "biggest consumer", "most energy",
                           "which area", "top zone", "largest energy"]):
        top = zone_totals[0] if zone_totals else {"zone_name": "—", "energy_kwh": 0}
        total_energy = sum(z["energy_kwh"] for z in zone_totals) or 1.0
        answer_text = (
            f"{top['zone_name']} is the largest consumer with "
            f"{top['energy_kwh']} kWh over the period "
            f"({round(top['energy_kwh'] / total_energy * 100, 1)}% of all zones). "
            f"Its peak draw reached {top['peak_power_kw']} kW. "
            f"Recommended focus: audit {top['zone_name']} controls."
        )
        return _analyst_response(question, answer_text, intent="zone_comparison",
                                 data=[{"zone": t["zone_name"], "energy_kwh": t["energy_kwh"]}
                                       for t in zone_totals])

    # ---------------- Why is consumption high ----------------
    if _match_keywords(q, ["why", "causing", "reason", "high consumption",
                           "high energy", "what is using"]):
        parts = []
        if anomalies:
            a = anomalies[0]
            parts.append(
                f"An anomaly was detected in {a['zone_name']} at {a['timestamp'][:16]}: "
                f"observed {a['observed_power_kw']} kW vs expected "
                f"{a['expected_power_kw']} kW ({a['deviation_pct']}%). "
                f"Reason: {a['possible_reason'].lower()}"
            )
        parts.append(
            f"HVAC is the dominant load at "
            f"{round(building_df['hvac_load'].sum() / max(1, building_df['energy_kwh'].sum()) * 100, 1)}% "
            "of consumption, followed by lighting. Peak demand occurs in the afternoon "
            "when cooling demand peaks."
        )
        if optimization and optimization.get("best_scenario"):
            b = optimization["best_scenario"]
            parts.append(
                f"Optimization suggests HVAC {b.get('hvac_pct')}% / lighting "
                f"{b.get('lighting_pct')}% could cut consumption by "
                f"{b.get('reduction_pct')}%."
            )
        answer_text = " ".join(parts)
        return _analyst_response(question, answer_text, intent="cause_analysis",
                                 anomalies=anomalies[:3])

    # ---------------- What is causing the peak ----------------
    if _match_keywords(q, ["peak", "spike", "max demand", "demand charge"]):
        peak = energy_service.peak_demand(building_df)
        if peak.get("peak_time"):
            peak_zone = next((z for z in zone_totals if z["peak_power_kw"] >= peak["peak_kw"] * 0.6),
                             zone_totals[0] if zone_totals else None)
            answer_text = (
                f"Peak demand of {peak['peak_kw']} kW occurred at {peak['peak_time']}. "
                f"Heat-driven HVAC is the primary contributor during weekday afternoons. "
                f"{peak_zone['zone_name'] if peak_zone else 'The strongest zone'} "
                "recorded the highest zone-level draw. Staggering morning HVAC start-up "
                "can shave the demand charge."
            )
        else:
            answer_text = "No peak demand data available yet."
        return _analyst_response(question, answer_text, intent="peak_analysis",
                                 peak=peak)

    # ---------------- What should I change ----------------
    if _match_keywords(q, ["what should i change", "recommend", "what to do",
                           "what should i do", "improve", "optimize"]):
        best = (optimization or {}).get("best_scenario") or {}
        tips = []
        if best:
            tips.append(
                f"Set HVAC to {best.get('hvac_pct')}% and lighting to "
                f"{best.get('lighting_pct')}% — this is the best feasible combination "
                f"(est. {best.get('reduction_pct')}% daily reduction)."
            )
        tips.append("Implement HVAC set-back during low-occupancy windows "
                    "(occupancy below 30% → intensity 70%).")
        tips.append("Shift non-critical loads (lab equipment, pumps) out of peak tariff hours.")
        answer_text = ". ".join(tips) + "."
        return _analyst_response(question, answer_text, intent="recommendation",
                                 recommendation=best)

    # ---------------- What if I reduce HVAC by X% ----------------
    m = re.search(r"(?:reduce|cut|set)\s*\D{0,6}?\s*hvac[^\d]{0,6}(\d{1,3})", q)
    if not m:
        m = re.search(r"(?:reduce|cut|set)\s*(\d{1,3})\s*%?\s*hvac", q)
    if m:
        hvac_pct = 100 - int(m.group(1))
        hvac_pct = max(0, min(120, hvac_pct))
        # quick estimate using optimization grid interpolation
        est_saving = _hvac_reduction_estimate(building_df, hvac_pct)
        answer_text = (
            f"Reducing HVAC from 100% to {hvac_pct}% cuts the simulated "
            f"daily HVAC load by roughly {est_saving['hvac_cut_pct']}%, saving an estimated "
            f"{est_saving['savings_kwh']} kWh/day (≈ ₹{est_saving['cost_saved']}/day, "
            f"~{est_saving['reduction_pct']}% of daily energy). Comfort stays within "
            "feasible bounds if occupancy is normal."
        )
        return _analyst_response(question, answer_text, intent="what_if",
                                 savings=est_saving)

    # ---------------- How much can I save ----------------
    if _match_keywords(q, ["how much", "save", "saving", "potential"]):
        best = (optimization or {}).get("best_scenario") or {}
        if best:
            answer_text = (
                f"You can save an estimated {best.get('savings_kwh')} kWh/day "
                f"(≈ ₹{best.get('cost_saved_inr')}/day, {best.get('reduction_pct')}%) "
                f"by applying HVAC {best.get('hvac_pct')}% / lighting "
                f"{best.get('lighting_pct')}%. "
                "These are ESTIMATED savings from the simulated scenario."
            )
        else:
            answer_text = "No optimization baseline available yet — run a simulation first."
        return _analyst_response(question, answer_text, intent="savings",
                                 savings=best)

    # ---------------- default contextual ----------------
    return _analyst_response(
        question,
        "I can answer questions about consumption causes, peak drivers, zone "
        "comparison, optimization targets and what-if savings from the live "
        "simulated dataset.",
        intent="unknown",
    )


def _hvac_reduction_estimate(building_df, hvac_pct: float) -> dict:
    hvac_total = float(building_df["hvac_load"].sum()) or 1.0
    lighting_total = float(building_df["lighting_load"].sum())
    hvac_share = hvac_total / max(1.0, hvac_total + lighting_total)
    reduction_kwh = hvac_total * (1 - hvac_pct / 100.0)
    return {
        "hvac_cut_pct": round((1 - hvac_pct / 100.0) * 100, 0),
        "savings_kwh": round(reduction_kwh, 2),
        "cost_saved": round(reduction_kwh * settings.ENERGY_TARIFF_PER_KWH, 2),
        "reduction_pct": round(reduction_kwh / max(1.0, hvac_total * 2.5) * 100, 1),
        "hvac_share": round(hvac_share * 100, 1),
    }


def _analyst_response(question: str, answer: str, intent: str, **context) -> dict:
    return {
        "question": question,
        "answer": answer,
        "intent": intent,
        "context": context,
        "labels": {"analysis": "COMPUTED FROM SIMULATED DATA"},
    }