from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolved relative to this file (not CWD) so it finds the repo-root .env
# regardless of whether the app is launched from backend/ or the repo root.
_REPO_ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_REPO_ROOT_ENV, extra="ignore")

    openai_api_key: str = ""
    google_places_api_key: str = ""
    openweather_api_key: str = ""
    osrm_base_url: str = "https://router.project-osrm.org"


@lru_cache
def get_settings() -> Settings:
    return Settings()
