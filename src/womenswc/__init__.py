from __future__ import annotations
from pathlib import Path

DATA_DIRECTORY = Path(__file__).parent.parent.parent.resolve() / "data"
HISTORICAL_DATA = DATA_DIRECTORY / "raw" / "odis_female_json"
WC_DATA = DATA_DIRECTORY / "raw" / "WC2025"
CITY_COUNTRY_CSV = DATA_DIRECTORY / "city_country.csv"
HISTORICAL_JSON_FILES = list(HISTORICAL_DATA.glob("*.json"))
