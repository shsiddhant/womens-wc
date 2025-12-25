from __future__ import annotations
from pathlib import Path

DATA_DIRECTORY = Path(__file__).parent.parent.parent.resolve() / "data"
HISTORICAL_DATA = DATA_DIRECTORY / "raw" / "2022-2025WODI"
WC_DATA = DATA_DIRECTORY / "raw" / "WC2025"
HISTORICAL_JSON_FILES = list(HISTORICAL_DATA.glob("*.json"))
