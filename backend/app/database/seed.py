"""Database seeding.

Creates the WattWise Demo Campus and populates it with simulated energy
data and baseline operating scenarios.
"""

from datetime import datetime, timedelta

from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.database.connection import Base, SessionLocal, engine
from app.database.models import (
    Building,
    EnergyReading,
    Equipment,
    Recommendation,
    Scenario,
    Zone,
)
from app.services.data_generator import (
    ZONE_CONFIG,
    ANOMALY_DEFS,
    generate_all_hourly,
    aggregate_building,
    generate_zone_hourly,
)

ZONE_NAMES = [z[0] for z in ZONE_CONFIG]
ANOMALY_ZONE_NAMES = {z[0] for z in ZONE_CONFIG if z[0] in {
    "Block A Classroom 1", "Block A Classroom 2", "Computer Lab", "Administration"
}}


def seed(force: bool = False) -> dict:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Building).count() > 0:
            if not force:
                latest_reading = db.query(EnergyReading.timestamp).order_by(
                    EnergyReading.timestamp.desc()
                ).first()
                if latest_reading:
                    latest_timestamp = latest_reading[0]
                    latest_demo_hour = datetime.now().replace(
                        minute=0, second=0, microsecond=0
                    ) - timedelta(hours=1)
                    if latest_timestamp < latest_demo_hour - timedelta(days=1):
                        shift = latest_demo_hour - latest_timestamp
                        for reading in db.query(EnergyReading).yield_per(1000):
                            reading.timestamp += shift
                        db.commit()
                        return {
                            "status": "ok",
                            "message": "Existing simulated readings shifted to the current demo period.",
                        }
                return {"status": "ok", "message": "Database already seeded. Use force=True to reseed."}
            db.query(Recommendation).delete()
            db.query(Scenario).delete()
            db.query(EnergyReading).delete()
            db.query(Equipment).delete()
            db.query(Zone).delete()
            db.query(Building).delete()
            db.commit()

        # 1) Building
        campus = Building(name="WattWise Demo Campus", location="Bengaluru, India", type="Campus")
        db.add(campus)
        db.commit()

        # 2) Zones + Equipment
        zone_obj_by_name = {}
        equipment_specs = {
            "classroom": [("HVAC Unit", "HVAC", "ON", 45.0), ("Lighting Bank", "Lighting", "ON", 12.0)],
            "computer-lab": [("HVAC Unit", "HVAC", "ON", 48.0), ("Server Racks", "Computers", "ON", 22.0),
                             ("Lighting Bank", "Lighting", "ON", 10.0)],
            "office": [("HVAC Unit", "HVAC", "ON", 32.0), ("Lighting Bank", "Lighting", "ON", 9.0)],
            "library": [("HVAC Unit", "HVAC", "ON", 28.0), ("Lighting Bank", "Lighting", "ON", 8.0)],
            "utility": [("Water Pump", "Pumps", "ON", 18.0), ("Booster Pump", "Pumps", "ON", 14.0)],
        }
        for name, base, hvac_w, ztype in ZONE_CONFIG:
            z = Zone(building_id=campus.id, name=name, zone_type=ztype,
                     area=round(base * 1.1, 1))
            db.add(z)
            db.flush()
            zone_obj_by_name[name] = z
            for (ename, etype, estatus, prating) in equipment_specs.get(ztype, []):
                db.add(Equipment(zone_id=z.id, name=ename, equipment_type=etype,
                                 status=estatus, power_rating=prating))
            if name == "Block A Classroom 1":
                db.add(Equipment(zone_id=z.id, name="Projector + Desktop", equipment_type="Computers",
                                 status="ON", power_rating=4.0))
        db.commit()

        # 3) Energy readings - 30 days of hourly simulated data
        start = datetime.now().replace(minute=0, second=0, microsecond=0) - timedelta(days=30)
        rows = generate_all_hourly(start, days=30)
        batch = []
        for r in rows:
            zi = r.pop("zone_index_offset")
            zname = ZONE_NAMES[zi]
            z = zone_obj_by_name[zname]
            batch.append(EnergyReading(
                timestamp=r["timestamp"],
                building_id=campus.id,
                zone_id=z.id,
                power_kw=r["power_kw"],
                energy_kwh=r["energy_kwh"],
                temperature=r["temperature"],
                occupancy=r["occupancy"],
                hvac_load=r["hvac_load"],
                lighting_load=r["lighting_load"],
                pump_load=r["pump_load"],
                other_load=r["other_load"],
            ))
            if len(batch) >= 5000:
                db.bulk_save_objects(batch)
                db.commit()
                batch = []
        if batch:
            db.bulk_save_objects(batch)
            db.commit()

        # 4) Baseline scenarios + recommendations
        baseline_energy = aggregate_building(rows, start, days=30)
        total_energy = sum(r["energy_kwh"] for r in baseline_energy)
        peak_power = max(r["power_kw"] for r in baseline_energy)

        # Average daily energy
        daily_energy = total_energy / 30.0

        scenario_energy = daily_energy * 0.90
        saving_energy = daily_energy - scenario_energy
        db.add(Scenario(
            name="Optimized HVAC + Lighting",
            hvac_percentage=78.0,
            lighting_percentage=85.0,
            occupancy_percentage=100.0,
            baseline_energy=round(daily_energy, 2),
            simulated_energy=round(scenario_energy, 2),
            estimated_savings=round(saving_energy, 2),
            estimated_cost_savings=round(saving_energy * settings.ENERGY_TARIFF_PER_KWH, 2),
        ))
        db.add(Scenario(
            name="HVAC High Performance",
            hvac_percentage=100.0,
            lighting_percentage=100.0,
            occupancy_percentage=100.0,
            baseline_energy=round(daily_energy, 2),
            simulated_energy=round(daily_energy, 2),
            estimated_savings=0.0,
            estimated_cost_savings=0.0,
        ))
        db.commit()

        # Recommendations derived from simulated patterns
        recs = [
            Recommendation(
                title="Reduce HVAC in Block A during low-occupancy hours",
                description="Set back HVAC intensity to 70% when zone occupancy is below 30% "
                            "between 10:00 AM and 4:00 PM.",
                reason="Consumption in Block A is unusually high while occupancy remains relatively low, "
                       "driving avoidable cooling load.",
                expected_savings=8.0,
                estimated_cost_saving=round(daily_energy * 0.08 * settings.ENERGY_TARIFF_PER_KWH, 2),
                confidence=0.91,
                feasibility="HIGH",
                status="OPEN",
            ),
            Recommendation(
                title="Shift Computer Lab after-hours loads",
                description="Schedule lab equipment shutdown at 6:00 PM unless a class is booked.",
                reason="Computer Lab shows elevated compute/cooling load on days 14-16 outside "
                       "operating hours.",
                expected_savings=5.0,
                estimated_cost_saving=round(daily_energy * 0.05 * settings.ENERGY_TARIFF_PER_KWH, 2),
                confidence=0.84,
                feasibility="MEDIUM",
                status="OPEN",
            ),
            Recommendation(
                title="Optimal HVAC start-time staggering",
                description="Stagger HVAC pre-cool across zones to reduce morning demand peak.",
                reason="Morning start-up creates a sharp demand spike that increases peak-demand cost.",
                expected_savings=4.0,
                estimated_cost_saving=round(daily_energy * 0.04 * settings.ENERGY_TARIFF_PER_KWH, 2),
                confidence=0.78,
                feasibility="HIGH",
                status="OPEN",
            ),
        ]
        db.add_all(recs)
        db.commit()

        return {
            "status": "ok",
            "message": f"Seeded: 1 building, {len(ZONE_NAMES)} zones, "
                       f"{len(rows)} hourly readings, {len(recs)} recommendations, 2 scenarios. "
                       f"Source: {settings.DATA_SOURCE_LABEL}.",
            "daily_energy_kwh": round(daily_energy, 2),
            "peak_power_kw": round(peak_power, 2),
        }
    except SQLAlchemyError as exc:
        db.rollback()
        return {"status": "error", "message": str(exc)}
    finally:
        db.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(seed(force=args.force))