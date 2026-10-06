from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)

from app.database.connection import Base


class Building(Base):
    __tablename__ = "buildings"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    location = Column(String, nullable=False)
    type = Column(String, nullable=False)


class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, index=True)
    building_id = Column(Integer, ForeignKey("buildings.id"), nullable=False)
    name = Column(String, nullable=False)
    zone_type = Column(String, nullable=False)
    area = Column(Float, default=0.0)


class EnergyReading(Base):
    __tablename__ = "energy_readings"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    building_id = Column(Integer, ForeignKey("buildings.id"), nullable=False)
    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=False)
    power_kw = Column(Float, nullable=False, default=0.0)
    energy_kwh = Column(Float, nullable=False, default=0.0)
    temperature = Column(Float, default=0.0)
    occupancy = Column(Float, default=0.0)
    hvac_load = Column(Float, default=0.0)
    lighting_load = Column(Float, default=0.0)
    pump_load = Column(Float, default=0.0)
    other_load = Column(Float, default=0.0)


class Equipment(Base):
    __tablename__ = "equipment"

    id = Column(Integer, primary_key=True, index=True)
    zone_id = Column(Integer, ForeignKey("zones.id"), nullable=False)
    name = Column(String, nullable=False)
    equipment_type = Column(String, nullable=False)
    status = Column(String, default="ON")
    power_rating = Column(Float, default=0.0)


class Scenario(Base):
    __tablename__ = "scenarios"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    hvac_percentage = Column(Float, default=100.0)
    lighting_percentage = Column(Float, default=100.0)
    occupancy_percentage = Column(Float, default=100.0)
    baseline_energy = Column(Float, default=0.0)
    simulated_energy = Column(Float, default=0.0)
    estimated_savings = Column(Float, default=0.0)
    estimated_cost_savings = Column(Float, default=0.0)


class Recommendation(Base):
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, default="")
    reason = Column(Text, default="")
    expected_savings = Column(Float, default=0.0)
    estimated_cost_saving = Column(Float, default=0.0)
    confidence = Column(Float, default=0.0)
    feasibility = Column(String, default="MEDIUM")
    status = Column(String, default="OPEN")