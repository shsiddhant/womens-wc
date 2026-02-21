from __future__ import annotations
import psycopg2.extras
import pandas as pd

def update_venue_map(conn: psycopg2.extensions.connection, city_country_csv: str):

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
            venue_name,
            city,
            country
            FROM cricket.venues
            WHERE city IS NOT NULL;
            """
        )
        c = cur.fetchall()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT
            venue_name,
            city,
            country
            FROM cricket.venues
            WHERE city IS NULL;
            """
        )
        c2 = cur.fetchall()
    # Update city values if NULL
    for venue in c2:
        if not venue["city"]:
            city = input(f'Venue: {venue["venue_name"]}\nCity: ')
            if city:
                venue["city"] = city

    df = pd.DataFrame(c)
    df_2 = pd.DataFrame(c2)

    df_3 = pd.concat([df, df_2], ignore_index=True)
    city_country = df_3[["city", "country"]]
    city_country.sort_values(by=["country", "city"], inplace=True, ignore_index=True)
    city_country.drop_duplicates(subset=["city"], inplace=True)
    c3 = city_country.to_dict(orient="records")
    # Update country if NULL
    for city in c3:
        if not city["country"]:
            city["country"] = input(f'City: {city["city"]}\nCountry: ')
    venue_df: pd.DataFrame
    venue_df = df_3[["venue_name", "city"]]
    city_country = pd.DataFrame(c3)
    df_main = venue_df.join(city_country.set_index("city"), on="city", validate="m:1")
    df_main.to_csv(city_country_csv, index=False, sep=";")

