from __future__ import annotations
from typing import TYPE_CHECKING
import json
import io
from pathlib import Path

if TYPE_CHECKING:
    import psycopg2

def city_country_table(
    conn: psycopg2.extensions.connection, city_country_csv: str | Path,
) -> None:
    """
    Copy city_country CSV to the database.

    conn : psycopg2.extensions.connection.
        A psycopg2 connection to the database.
    city_country_csv : str, Path
        The path to CSV containing the city_country dictionary.

    """
    with open(city_country_csv, "r") as file:
        with conn.cursor() as cur:
            cur.copy_from(
            file, "city_country", sep=";", null="", columns=("city", "country", "venue")
            )
    conn.commit()

def copy_json_to_table(
    conn: psycopg2.extensions.connection,
    json_files: list[str | Path],
    json_table_name: str
) -> None:
    """
    Copy JSON files as JSONB rows in a temp table.

    conn : psycopg2.extensions.connection.
        A psycopg2 connection to the database.
    json_files : list[str | Path]
        A list containing the paths to the match JSON files.
    json_table_name : str
        Name of the staging table to copy JSON data to.

    """
    for match_json in json_files:
        match_json = Path(match_json).resolve()
        with open(match_json, "r") as file:
            data = json.load(file)
        data["match_id"] = int(match_json.name.removesuffix(".json"))
        with conn.cursor() as cur:
            cur.copy_expert(
            f"COPY {json_table_name} (data) FROM STDIN",
            io.StringIO(json.dumps(data))
            )
    conn.commit()
