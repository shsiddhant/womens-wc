# Weighted Cumulative Sums
# -------------------------------------------------------------

from __future__ import annotations
from typing import Literal
import pandas as pd
import numpy as np


# Sorted list of unique teams
def teams_list(base_df: pd.DataFrame) -> np.ndarray:
    teams = pd.unique(
        pd.concat([base_df.team_0, base_df.team_1], axis=0, ignore_index=True)
    )
    return teams


# Exponential decay weights
def expweights_list(base_df: pd.DataFrame, half_life=180) -> pd.Series:
    k = 1 / half_life
    date_min = base_df.start_date.min()
    days = (base_df.start_date - date_min).dt.days
    return pd.Series(
        np.exp2(-k * days), index=base_df.index, name="exponential decay weight"
    )


# Custom cumulative sum (Exclude current row)
def custom_cumsum(x: pd.Series):
    return x.cumsum() - x


# Weighted cumulative sum team array(Exclude current row)
def weighted_cumsum(
    ser_0: pd.Series,
    ser_1: pd.Series,
    weight: pd.Series,
    teams: list[str],
    is_team_0: dict[str, pd.Series],
    is_team_1: dict[str, pd.Series],
) -> np.ndarray:
    """
    Calculate weighted cumulative sums for numeric features.

    Parameters
    ----------
    ser_0 : pd.Series
        The base_dataset column with values for team_0. For instance, base_df.runs_0.
    ser_1 : pd.Series
        The base_dataset  column with values for team_1. It must be of the same
        kind.
        For instance, if ser_0 = base_df.runs_0, then ser_1 must be base_df.runs_1.
    weight : pd.Series
        A pandas series containing weights.
    is_team_0 : dict[str, pd.Series]
        A dictionary containing all characteristic function values for each team over
        column team_0. For each team, is_team_0 is a boolean series, which is True for
        matches when the team played as team_0.
        For example: ``is_team["India"] = (base_df.team_0 == "India")``
     is_team_1 : dict[str, pd.Series]
        Exactly like is_team_0 except it's for team_1.

    Note: All four must have the same index.

    Returns
    -------
    np.ndarray
        A numpy array containing weighted cumulative sums for the feature for each team.
        Rows are matches, and columns are distinct teams (sorted).
    """
    ser = {team: ser_0 * is_team_0[team] + ser_1 * is_team_1[team] for team in teams}

    wt_cumsum = np.array(
        [weight * custom_cumsum(ser[team] / weight) for team in teams]
    ).transpose()
    return wt_cumsum


# Get Indexer for NumPy broadcasting and advanced indexing.
def indexer(base_df: pd.DataFrame, teams: list[str], n: Literal[0, 1]):
    return (np.arange(len(base_df)), pd.Index(teams).get_indexer(base_df[f"team_{n}"]))


# Weighted Cumulative Sum for any numeric column
def weighted_cumsum_column(
    base_df: pd.DataFrame,
    column: str,
    weight: pd.Series,
    teams: list[str],
    is_team_0: dict[str, pd.Series],
    is_team_1: dict[str, pd.Series],
    is_bowling_side: bool = True,
    n: Literal[0, 1] = 0,  # 0 for team_0 and 1 for team_1
) -> np.ndarray:
    m = int(is_bowling_side)
    # For bowling side stats, 0 and 1 are switched.
    # For example, if column = 'runs', then if,
    # 1. is_bowling_side == True, the weighted cumsum is for runs conceded.
    # 2. is_bowling_side == False, the weighted cumsum is for runs scored.
    if n in [0, 1]:
        return weighted_cumsum(
            base_df[f"{column}_{m}"],
            base_df[f"{column}_{1 - m}"],
            weight,
            teams,
            is_team_0,
            is_team_1,
        )[indexer(base_df, teams, n)]
    else:
        raise ValueError("'n' must either be 0 or 1")


def weighted_agg_stats(
    base_df: pd.DataFrame,
    weight: pd.Series,
    teams: list[str],
    is_team_0: dict[str, pd.Series],
    is_team_1: dict[str, pd.Series],
) -> dict[str, np.ndarray]:
    weighted_stats = {}
    batting_side_dict = {
        "runs": "runs_scored",
        "wickets": "wickets_lost",
        "deliveries": "deliveries_played",
    }
    bowling_side_dict = {
        "runs": "runs_conceded",
        "wickets": "wickets_taken",
        "deliveries": "deliveries_bowled",
    }

    # Weighted Cumulative Runs Scored, Wickets Lost, Deliveries Played
    for n in [0, 1]:
        for column, stat in batting_side_dict.items():
            weighted_stats[f"{stat}_{n}"] = weighted_cumsum_column(
                base_df,
                column,
                weight,
                teams,
                is_team_0,
                is_team_1,
                is_bowling_side=False,
                n=n,
            )
        # Weighted Cumulative Runs Conceded, Wickets Taken, Deliveries bowled
        for column, stat in bowling_side_dict.items():
            weighted_stats[f"{stat}_{n}"] = weighted_cumsum_column(
                base_df,
                column,
                weight,
                teams,
                is_team_0,
                is_team_1,
                is_bowling_side=True,
                n=n,
            )
        # Weighted Win Count
        weighted_stats[f"wins_{n}"] = weighted_cumsum(
            base_df.result == 0,  # when summed, you get win count
            base_df.result == 1,
            weight,
            teams,
            is_team_0,
            is_team_1,
        )[indexer(base_df, teams, n)]
        # Weighted Match Count
        weighted_stats[f"matches_{n}"] = weighted_cumsum(
            np.ones(
                shape=len(base_df),
            ),  # when summed, you get match count
            np.ones(
                shape=len(base_df),
            ),
            weight,
            teams,
            is_team_0,
            is_team_1,
        )[indexer(base_df, teams, n)]

    return weighted_stats


def drop_zeros_in_denominator(
    base_df: pd.DataFrame,
    weight: pd.Series,
    teams: list[str],
    is_team_0: dict[str, pd.Series],
    is_team_1: dict[str, pd.Series],
):
    # Initialize filtering condition
    condition = [False] * len(base_df)
    for n in [0, 1]:
        weighted_stats = weighted_agg_stats(
            base_df, weight, teams, is_team_0, is_team_1
        )
        a = weighted_stats[f"wickets_lost_{n}"]  # Wickets Lost
        b = weighted_stats[f"deliveries_played_{n}"]  # Deliveries Played
        # Wickets Taken (Switch 0 and 1)
        c = weighted_stats[f"wickets_taken_{n}"]
        # Deliveries Bowled (Switch 0 and 1)
        d = weighted_stats[f"deliveries_bowled_{n}"]
        e = weighted_stats[f"matches_{n}"]  # Matches Played
        condition = condition | (a == 0) | (b == 0) | (c == 0) | (d == 0) | (e == 0)
    return ~condition
