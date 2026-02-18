from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import pandas as pd
import psycopg2.extras
from pathlib import Path

if TYPE_CHECKING:
    import datetime

def features (
    conn:psycopg2.extensions.connection,
    teams: Iterable[str],
    features_script: str | Path,
    cutoff_date: str | pd.Timestamp | datetime.date,
    include_tie_no_result: bool = False,
    ):
    """
    Build features from database.

    conn : psycopg2.extensions.connection
        A psycopg2 connection to the database.
    teams : A sequence of strings
        The sequence of teams to consider for feature building.
    features_script : str, pathlib Path
        Path to the features_script.
    cutoff_date : str, pd.Timestamp, datetime.date
        A minimum cutoff date for matches. Any match before the cutoff is filtered
        out while building features.

    """
    cutoff_date = pd.Timestamp(cutoff_date).normalize()
    for team in teams:
        team.capitalize()
    teams = tuple(teams)
    if len(teams) == 0:
        raise ValueError("Sequence must have atleast one value")
    with open(features_script, "r") as file:
        features_sql = file.read()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                features_sql,
                (include_tie_no_result,
                 teams, teams, cutoff_date.date(),
                 include_tie_no_result,
                 teams, teams, cutoff_date.date())
            )
            conn.commit()
    except conn.Error as e:
        conn.rollback()
        raise conn.Error(e)

def snapshots(
    conn:psycopg2.extensions.connection,
    teams: Iterable[str],
    features_script: str | Path,
    cutoff_date: str | pd.Timestamp | datetime.date,
    include_tie_no_result: bool = False,
    save_dir = str | Path
    ):
    """
    Build and export features and target as CSV.

    conn : psycopg2.extensions.connection
        A psycopg2 connection to the database.
    teams : A sequence of strings
        The sequence of teams to consider for feature building.
    features_script : str, pathlib Path
        Path to the features_script.
    cutoff_date : str, pd.Timestamp, datetime.date
        A minimum cutoff date for matches. Any match before the cutoff is filtered
        out while building features.
    save_dir : str, pathlib Path
        Path to the directory where snapshots are to be saved.

    """
    features(conn, teams, features_script, cutoff_date)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM features ft JOIN target t ON t.match_id = ft.match_id"
            )
            df = pd.DataFrame(cur.fetchall())
    except psycopg2.Error as e:
        raise psycopg2.Error(e)
    features_df: pd.DataFrame
    features_df = df.loc[:, :"result"]
    target_df = df[["match_id", "result"]]
    path = Path(save_dir)
    features_df.to_csv(path / f"features_{cutoff_date}.csv", index=False)
    target_df.to_csv(path / f"target_{cutoff_date}.csv", index=False)

