from __future__ import annotations
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
import os

from womenswc import DATA_DIRECTORY, HISTORICAL_JSON_FILES, CITY_COUNTRY_CSV
from womenswc.copy_from_files import (
    city_country_table,
    copy_json_to_table,
    copy_deliveries_json,
)

def main():
    json_files = HISTORICAL_JSON_FILES
    print("JSON files:", json_files)
    load_dotenv(DATA_DIRECTORY.parent / "config" / ".env")
    conn: psycopg2.extensions.connection
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASS"),
            host=os.getenv("DB_HOST")
        )
        init_db(conn, schema="womenswc")
        city_country_table(conn, city_country_csv=CITY_COUNTRY_CSV)
        copy_json_to_table(
            conn, json_files, schema="womenswc", json_table_name="raw_json"
        )
        copy_deliveries_json(conn, json_files, schema="womenswc")
    except psycopg2.Error as e:
        print("Error:", e)
        conn.rollback()
    else:
        conn.commit()

def init_db(
    conn: psycopg2.extensions.connection,
    schema: str = "womenswc",
    incremental: bool = False,
):
    drop_schema_sql = (
        f"""
        DROP SCHEMA IF EXISTS {schema} CASCADE;
        CREATE SCHEMA {schema};
        """
    )
    create_tables = (
        f"""
        CREATE TABLE {schema}.raw_json (
            id SERIAL,
            data JSONB,
            PRIMARY KEY (id)
        );
        CREATE TABLE {schema}.deliveries_json (
            match_id INT,
            hash_id TEXT,
            n_innings INT,
            team TEXT,
            n_over INT,
            n_delivery INT,
            delivery JSONB
        );
        CREATE TABLE {schema}.city_country (
            id SERIAL,
            city TEXT,
            country TEXT,
            venue_name TEXT,
            UNIQUE (city, venue_name),
            PRIMARY KEY (id)
        );
        """
    )
    if not incremental:
        with conn.cursor() as cur:
            cur.execute(drop_schema_sql)
            cur.execute(create_tables)

if __name__ == "__main__":
    main()
