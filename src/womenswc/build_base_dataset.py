from __future__ import annotations
from typing import TYPE_CHECKING

import json
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

from womenswc import (
    DATA_DIRECTORY,
    HISTORICAL_DATA,
)
from womenswc.match_data_util import (
    toss,
    get_scores,
    results,
)


# Parse Match Data
def matchdict(
    match_data: dict, match_id: str, city_to_country: dict
) -> dict[str, int | str | None]:
    teams = match_data["info"]["teams"]
    teams.sort()
    if "winner" not in match_data["info"]["outcome"].keys():
        return {"match_id": match_id, "result": match_data["info"]["outcome"]["result"]}
    toss_data = toss(match_data)
    scores = get_scores(match_data)
    return {
        "match_id": match_id,
        "country": city_to_country[match_data["info"]["city"]],
        "start_date": match_data["info"]["dates"][0],
        "event": match_data["info"]["event"]["name"],
        "team_0": teams[0],
        "team_1": teams[1],
        "toss_winner": toss_data[0],
        "toss_decision": toss_data[1],
        "runs_0": scores[0]["runs"],
        "wickets_0": scores[0]["wickets"],
        "deliveries_0": 6 * scores[0]["overs"][0] + scores[0]["overs"][1],
        "runs_1": scores[1]["runs"],
        "wickets_1": scores[1]["wickets"],
        "deliveries_1": 6 * scores[1]["overs"][0] + scores[1]["overs"][1],
        "result": results(match_data),
    }


# Read match data from JSON
def get_match_data_(
    match_json: Path, city_to_country: dict
) -> dict[str, int | str | None]:
    match_id = match_json.name.removesuffix(".json")
    with open(match_json, "r") as fp:
        return matchdict(json.load(fp), match_id, city_to_country)


# Build Dataset
def build_db(
    get_match_data_vec: Callable,
    city_to_country,
    historial_datadir: Path = HISTORICAL_DATA,
):
    json_files = list(historial_datadir.glob("*.json"))
    return pd.DataFrame.from_records(
        get_match_data_vec(json_files, city_to_country)
    ).dropna()


def main():
    with open(DATA_DIRECTORY / "city-to-country.json", "r") as fp:
        city_to_country = json.load(fp)
    get_match_data_vec = np.vectorize(get_match_data_, otypes=[dict])
    base_df = build_db(get_match_data_vec, city_to_country)
    base_df["start_date"] = pd.to_datetime(base_df.start_date)
    base_df = base_df.sort_values(by=["start_date"])
    base_df = base_df[base_df.start_date.dt.year >= 2022]
    base_df = base_df.reset_index()
    base_df = base_df.drop(columns=["index"])
    base_df.to_parquet(DATA_DIRECTORY / "processed" / "base_dataset.parquet")


if __name__ == "__main__":
    main()
