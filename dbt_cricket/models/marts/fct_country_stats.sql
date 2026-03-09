
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id']
    )
}}

------------------------------------------------

WITH

matches AS (

    SELECT * FROM {{ ref('dbt_cricket', 'matches') }}

),

match_teams AS (

    SELECT * FROM {{ ref('dbt_cricket', 'match_teams') }}

),

venues AS (

    SELECT * FROM {{ ref('dbt_cricket', 'venues') }}

),

country_stats AS (

    SELECT

        match_teams.match_id,
        match_teams.hash_id,
        venues.country,

        ROUND(
            1.0 * SUM(1 - match_teams.batted_first::int) OVER w / COUNT(*) OVER w, 2
        ) AS chasing_advantage

    FROM match_teams

    RIGHT JOIN matches
        ON matches.match_id = match_teams.match_id

    JOIN venues
        ON venues.venue_id = matches.venue_id

    WHERE match_teams.won_match

    WINDOW w AS (
        PARTITION BY venues.country
        ORDER BY matches.start_date
        ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW
    )

)

SELECT * FROM country_stats      

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
