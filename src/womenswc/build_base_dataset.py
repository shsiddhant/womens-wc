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
def get_match_data(
    match_json: Path, city_to_country: dict
) -> dict[str, int | str | None]:
    match_id = match_json.name.removesuffix(".json")
    with open(match_json, "r") as fp:
        return matchdict(json.load(fp), match_id, city_to_country)


# Build Dataset
def build_db(
    get_match_data: Callable,
    city_to_country,
    historical_datadir: Path = HISTORICAL_DATA,
):
    json_files = list(historical_datadir.glob("*.json"))
    base_df = pd.DataFrame.from_records(
        [get_match_data(match_json, city_to_country) for match_json in json_files]
    ).dropna()
    base_df["start_date"] = pd.to_datetime(base_df.start_date)
    base_df = base_df.sort_values(by=["start_date"])
    base_df = base_df[base_df.start_date.dt.year >= 2022]
    base_df = base_df.reset_index(drop=True)
    return base_df

def swap_labels(base_df: pd.DataFrame, seed: int = 1712):
    """
    Randomly swap labels to avoid bias in modeling, which
    is expected since Australia are always first alphabetically,
    and are the strongest team amongst women's cricket teams
    by a fair margin.
    """
    df = base_df.copy()
    rng = np.random.default_rng(seed=seed)
    mask = pd.Series(rng.choice([True, False], size=len(df)), index=df.index)
    for col in df.columns:
        if col.endswith("_0"):
            col_1 = col[:-2] + "_1"
            old_col_0 = df[col].copy()
            df.loc[mask, col] = df.loc[mask, col_1]
            df.loc[mask, col_1] = old_col_0.loc[mask]
    df.loc[mask, "result"] = 1 - df.loc[mask, "result"]
    return df

def main():
    with open(DATA_DIRECTORY / "city-to-country.json", "r") as fp:
        city_to_country = json.load(fp)
    base_df = build_db(get_match_data, city_to_country)
    base_df = swap_labels(base_df, 171205121729)
    base_df.to_parquet(DATA_DIRECTORY / "processed" / "base_dataset.parquet")
    print(
        "Base dataset created and saved to "
        f"{DATA_DIRECTORY / 'processed' / 'base_dataset.parquet'}"
    )


if __name__ == "__main__":
    main()
