from __future__ import annotations
from typing import TYPE_CHECKING
from womenswc.copy_from_files import (
    city_country_table,
    copy_json_to_table,
    copy_deliveries_json,
)

if TYPE_CHECKING:
    import psycopg2
    from pathlib import Path


def init_db(
    conn: psycopg2.extensions.connection,
    init_path: Path | None = None,
    *,
    init_command: str | None = None,
    ) -> None:
    """
    Initialize the database from schema.

    conn : psycopg2.extensions.connection
        A psycopg2 connection to the database.
    init_path : str, Path, Optional
        The path to the initializing sql script.
    init_command : str, Optional
        The commands to initialize the database.
        Only one of ``init_path`` or ``init_command`` must be passed.

    """
    if init_path and init_command:
        raise ValueError(
            "Only one of 'init_path' and 'init_command' is allowed."
            )
    elif init_command:
        with conn.cursor() as cur:
            cur.execute(init_command)
        conn.commit()
    elif init_path:
        with open(init_path, "r") as file:
            with conn.cursor() as cur:
                cur.execute(file.read())
        conn.commit()
    else:
        raise ValueError("No commands or path provided")

def insert_from_json(
    conn: psycopg2.extensions.connection,
    json_files: list[str | Path],
    city_country_csv: str | Path,
    insert_data_script: str | Path,
) -> None:
    """
    Insert the data from JSON into database tables.

    conn : psycopg2.extensions.connection.
    json_files : list containing str or Path objects.
        A list containing the paths to the match JSON files.
        A psycopg2 connection to the database.
    city_country_csv : str, Path
        The path to CSV containing the city_country dictionary.
    insert_sql_script : str, Path
        The path to the SQL script for inserting data into database tables.

    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
            match_id::text
            FROM matches
            """)
        existing_id_list = cur.fetchall()
        cur.execute(
            """
            CREATE TEMPORARY TABLE json_table (
            id SERIAL,
            data JSONB,
            PRIMARY KEY (id)
        );
            """)
    for match_json in json_files:
        if (match_json.name.removesuffix(".json"),) in existing_id_list:
            json_files.remove(match_json)
    copy_json_to_table(conn, json_files, "json_table")
    city_country_table(conn, city_country_csv)
    copy_deliveries_json(conn, json_files)
    with open(insert_data_script, "r") as file:
        sql = file.read()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
    except conn.Error as e:
        conn.rollback()
        print(e)
        raise conn.Error(e)


def update_db(
    conn: psycopg2.extensions.connection,
    json_files,
    city_country_csv,
    insert_data_script: str | Path,
):
    """
    Update database.

    conn : psycopg2.extensions.connection.
    json_files : list containing str or Path objects.
        A list containing the paths to the match JSON files.
        A psycopg2 connection to the database.
    city_country_csv : str, Path
        The path to CSV containing the city_country dictionary.
    insert_sql_script : str, Path
        The path to the SQL script for inserting data into database tables.

    """

    with conn:
        insert_from_json(conn, json_files, city_country_csv, insert_data_script)

def reinit_db(
    conn: psycopg2.extensions.connection,
    init_path,
    json_files,
    city_country_csv,
    insert_data_script: str | Path,
):
    """
    Rebuild database from scratch.
    """

    try:
        with conn:
            cur = conn.cursor()
            cur.execute("DROP SCHEMA public CASCADE;")
            cur.execute("CREATE SCHEMA public;")
            init_db(conn, init_path=init_path)
        insert_from_json(conn, json_files, city_country_csv, insert_data_script)
        conn.commit()
    except conn.Error as e:
        conn.rollback()
        raise conn.Error(e)
