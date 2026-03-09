
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'innings_number']
    )
}}

------------------------------------------------

WITH

matches AS (

    SELECT * FROM {{ ref('dbt_cricket', 'matches') }}

),

deliveries AS (
    
    SELECT * FROM {{ ref('dbt_cricket', 'deliveries') }}

),

match_scorecard AS (

    SELECT

        deliveries.match_id,
        deliveries.innings_number,
        
        ANY_VALUE(matches.start_date) AS start_date,
        ANY_VALUE(matches.hash_id) AS hash_id,
        ANY_VALUE(matches.venue_id) AS venue_id,
        ANY_VALUE(deliveries.team) AS team,

        SUM(deliveries.runs) AS runs,
        SUM(deliveries.extras) AS extras,

        COUNT(deliveries.player_out) FILTER (WHERE deliveries.player_out IS NOT NULL)
            AS wickets

    FROM matches

    LEFT JOIN deliveries
        ON matches.match_id = deliveries.match_id

    GROUP BY deliveries.match_id, deliveries.innings_number
       
)

SELECT * FROM match_scorecard

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
