from __future__ import annotations
from typing import TYPE_CHECKING
import json
import io
import csv
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
            cur.execute(
                """
                CREATE TEMPORARY TABLE city_country_temp (
                city TEXT,
                country TEXT,
                venue TEXT,
                UNIQUE (city, venue)
                );
                """
            )
            cur.copy_from(
                file, "city_country_temp", sep=";", null="",
                columns=("city", "country", "venue")
            )
            cur.execute(
                """
                INSERT INTO city_country (
                city,
                venue,
                country
                )
                SELECT DISTINCT ON (city, venue)
                city,
                venue,
                country
                FROM city_country_temp
                ON CONFLICT (city, venue) DO NOTHING;
                """
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

def json_explode(json_file_path, deliveries=[]):
    with open(json_file_path, "r") as file:
        data = json.load(file)
    for n, inn in enumerate(data["innings"]):
        for o, over in enumerate(inn["overs"]):
            for n_d, deliv in enumerate(over["deliveries"]):
                d = {}
                d["match_id"] = int(
                    Path(json_file_path).name.removesuffix(".json")
                )
                d["n_innings"] = n
                d["team"] = inn["team"]
                d["n_over"] = o
                d["n_delivery"] = n_d
                d["delivery"] = json.dumps(deliv)
                deliveries.append(d)
    return deliveries

def copy_deliveries_json(
    conn: psycopg2.extensions.connection,
    json_files
) -> None:
    """
    """
    columns = [
        "match_id",
        "n_innings",
        "team",
        "n_over",
        "n_delivery",
        "delivery",
    ]
    deliveries = []
    stdin = io.StringIO()
    writer = csv.DictWriter(stdin, fieldnames=columns)
    for match_json in json_files:
        deliveries = json_explode(match_json, deliveries)
    writer.writerows(deliveries)
    stdin.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(
            f"COPY deliveries_json ({','.join(columns)}) FROM STDIN WITH CSV",
            stdin
        )
