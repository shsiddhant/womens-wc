
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'innings_number']
    )
}}

------------------------------------------------

WITH

deliveries_staging AS (

    SELECT * FROM {{ ref('dbt_cricket', 'stg_deliveries') }}

),

innings AS (

    SELECT

        match_id,
        innings_number,
        MAX(hash_id) AS hash_id,
        MAX(team) AS team,
        SUM(runs) AS runs_scored,
        COUNT(player_out) AS wickets_lost,
        COUNT(*) FILTER (WHERE is_legal) AS legal_deliveries,
        SUM(wides) AS wides,
        SUM(noballs) AS noballs,
        SUM(byes) AS byes,
        SUM(legbyes) AS legbyes,
        SUM(extras) AS extras

    FROM deliveries_staging 

    GROUP BY (match_id, innings_number)

)

SELECT * FROM innings

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
