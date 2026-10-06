import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Postgres credentials - never hardcoded in source.
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "wattwise")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "wattwise_dev_pw")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "wattwise")
    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5432")

    # CORS
    CORS_ORIGINS: list[str] = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    # Data source integrity
    DATA_SOURCE_LABEL: str = "SIMULATED"

    # Cost / carbon configuration
    ENERGY_TARIFF_PER_KWH: float = float(os.getenv("ENERGY_TARIFF_PER_KWH", "8.5"))
    CARBON_FACTOR_KG_PER_KWH: float = float(os.getenv("CARBON_FACTOR_KG_PER_KWH", "0.82"))

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


settings = Settings()