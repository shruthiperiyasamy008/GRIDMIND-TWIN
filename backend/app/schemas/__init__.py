from datetime import datetime

from pydantic import BaseModel, Field, field_validator


class ZoneControl(BaseModel):
    hvac: float = Field(100.0, ge=0, le=120)
    lighting: float = Field(100.0, ge=0, le=100)
    occupancy: float = Field(100.0, ge=0, le=100)

    @field_validator("hvac")
    @classmethod
    def hvac_range(cls, v):
        if (v < 0 or v > 120):
            raise ValueError("hvac must be between 0 and 120")
        return v


class SimulateRequest(BaseModel):
    zone: str = Field(...)
    hvac: float = Field(100.0, ge=0, le=120)
    lighting: float = Field(100.0, ge=0, le=100)
    occupancy: float = Field(100.0, ge=0, le=100)


class BuildingSimulateRequest(BaseModel):
    zones: dict[str, ZoneControl] = Field(default_factory=dict)


class ScenarioCreate(BaseModel):
    name: str = Field(...)
    hvac_percentage: float = Field(100.0, ge=0, le=120)
    lighting_percentage: float = Field(100.0, ge=0, le=100)
    occupancy_percentage: float = Field(100.0, ge=0, le=100)
    baseline_energy: float = Field(0.0, ge=0)
    simulated_energy: float = Field(0.0, ge=0)
    estimated_savings: float = Field(0.0)
    estimated_cost_savings: float = Field(0.0)


class AnalystRequest(BaseModel):
    question: str = Field(..., min_length=2)


class ForecastRequest(BaseModel):
    horizon_hours: int = Field(24, ge=1, le=168)


class AnalyticsQuery(BaseModel):
    granularity: str = Field("daily", pattern="^(hourly|daily|weekly|monthly)$")
    building_id: int | None = None
    zone_id: int | None = None
    start: datetime | None = None
    end: datetime | None = None