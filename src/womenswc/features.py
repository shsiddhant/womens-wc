from __future__ import annotations
from typing import Literal, TYPE_CHECKING
import json
import pandas as pd
from womenswc import DATA_DIRECTORY
from womenswc.weights_util import (
    expweights_list,
    weighted_agg_stats,
    drop_zeros_in_denominator,
)

if TYPE_CHECKING:
    import numpy as np

# Features
# -------------------------------------------------------------------

# 1. Home Advantage
# -------------------------------------------------------------------


def home_advantage(base_df: pd.DataFrame, n: Literal[0, 1]) -> pd.Series:
    """
    Home Advantage for team_n.

    Equals 1 if team_n is home team, -1 if away team, and 0 if neutral.
    """
    m = int(not n)
    return (base_df[f"team_{n}"] == base_df["country"]).astype(int) - (
        base_df[f"team_{m}"] == base_df["country"]
    ).astype(int)


# Weighted Features
# -------------------------------------------------------------------
# Win Percentage
# -------------------------------------------------------------------
def win_percentage(
    weighted_stats, n: Literal[0, 1], filter_zeros: pd.Series
) -> pd.Series:
    """
    Weighted Win Percentage of team_n.

    """
    return (
        100
        * (weighted_stats[f"wins_{n}"][filter_zeros])
        / (weighted_stats["matches_{n}"][filter_zeros])
    )


# Batting Average
# -------------------------------------------------------------------
def batting_average(
    weighted_stats, n: Literal[0, 1], filter_zeros: pd.Series
) -> pd.Series:
    """
    Weighted Batting Average of team_n.

    Batting Average = Runs scored / Wickets Lost

    """
    return (
        weighted_stats[f"runs_scored_{n}"][filter_zeros]
        / weighted_stats[f"wickets_lost_{n}"][filter_zeros]
    )


# Batting Strike-rate
# -------------------------------------------------------------------
def batting_strike_rate(
    weighted_stats, n: Literal[0, 1], filter_zeros: pd.Series
) -> pd.Series:
    """
    Weighted Batting Strike-rate of team_n.

    Batting Strike-rate = 100 * (Runs scored / Bowls faced)

    """
    return (
        100
        * weighted_stats[f"runs_scored_{n}"][filter_zeros]
        / weighted_stats[f"deliveries_played_{n}"][filter_zeros]
    )


# Bowling Average
# -------------------------------------------------------------------
def bowling_average(
    weighted_stats, n: Literal[0, 1], filter_zeros: pd.Series
) -> pd.Series:
    """
    Weighted Bowling Average of team_n.

    Bowling Average = Runs conceded / Wickets taken

    """
    return (
        weighted_stats[f"runs_conceded_{n}"][filter_zeros]
        / weighted_stats[f"wickets_taken_{n}"][filter_zeros]
    )


# Bowling Economy
# -------------------------------------------------------------------
def bowling_economy(
    weighted_stats, n: Literal[0, 1], filter_zeros: pd.Series
) -> pd.Series:
    """
    Weighted Bowling Economy of team_n.

    Bowling Economy = 6 * (Runs conceded / Deliveries bowled)

    """
    return (
        6
        * weighted_stats[f"runs_conceded_{n}"][filter_zeros]
        / weighted_stats[f"deliveries_bowled_{n}"][filter_zeros]
    )


def build_features(
    base_df: pd.DataFrame, teams: list[str], weight: pd.Series | np.ndarray
) -> pd.DataFrame:
    # Mask
    is_team_0 = {team: (base_df.team_0 == team) for team in teams}
    is_team_1 = {team: (base_df.team_1 == team) for team in teams}
    # Weighted Aggregate Stats
    weighted_stats = weighted_agg_stats(base_df, weight, teams, is_team_0, is_team_1)
    # Filter out rows with zero in any column that is used to divide another column
    filter_zeros = drop_zeros_in_denominator(
        base_df, weight, teams, is_team_0, is_team_1
    )
    df: pd.DataFrame
    df = base_df[["team_0", "team_1"]][filter_zeros].copy()
    for n in [0, 1]:
        df[f"home_adv_{n}"] = home_advantage(base_df, n)[filter_zeros]
        df[f"batting_average_{n}"] = batting_average(weighted_stats, n, filter_zeros)
        df[f"batting_sr_{n}"] = batting_strike_rate(weighted_stats, n, filter_zeros)
        df[f"bowling_average_{n}"] = bowling_average(weighted_stats, n, filter_zeros)
        df[f"bowling_economy_{n}"] = bowling_economy(weighted_stats, n, filter_zeros)
    df = df.round(decimals=2)


def main():
    base_df = pd.read_parquet(DATA_DIRECTORY / "processed" / "base_dataset.parquet")
    with open("./data/teams_list.json") as fp:
        teams = json.load(fp)
    weight = expweights_list(base_df, half_life=180)
    df = build_features(base_df, teams, weight)
    df.to_parquet(DATA_DIRECTORY / "processed" / "features_dataset.parquet")
    print(
        "Features dataset created and saved to "
        f"{DATA_DIRECTORY / 'processed' / 'features_dataset.parquet'}"
    )


if __name__ == "__main__":
    main()
