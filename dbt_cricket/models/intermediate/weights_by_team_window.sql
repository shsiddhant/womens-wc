
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'team']
    )
}}

------------------------------------------------

{% set half_life = 30 %}

WITH

matches AS (

    SELECT * FROM {{ ref('dbt_cricket', 'matches') }}

),

match_teams AS (

    SELECT * FROM {{ ref('dbt_cricket', 'match_teams') }}

),

matches_count AS (

    SELECT

        matches.match_id,
        matches.hash_id,
        matches.start_date,
        match_teams.team,
        COUNT(*) over w AS n_matches

    FROM matches

    JOIN match_teams

    ON match_teams.match_id = matches.match_id

    window w AS (
        PARTITION BY match_teams.team
        ORDER BY matches.start_date
        ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW
    )

),

weights_by_team_window AS (

    SELECT

        match_id,
        hash_id,
        start_date,
        team,
        n_matches,
        EXP(- LN(2) * n_matches / '{{ half_life }}') AS weight

    FROM matches_count

)

SELECT * FROM weights_by_team_window

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
