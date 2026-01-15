from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import pandas as pd
import psycopg2.extras

if TYPE_CHECKING:
    import datetime
    from pathlib import Path

def features (
    conn:psycopg2.extensions.connection,
    teams: Iterable[str],
    features_script: str | Path,
    cutoff_date: str | pd.Timestamp | datetime.date,
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
                (teams, teams, cutoff_date.date(), teams, teams, cutoff_date.date())
            )
            conn.commit()
    except conn.Error as e:
        conn.rollback()
        raise conn.Error(e)

