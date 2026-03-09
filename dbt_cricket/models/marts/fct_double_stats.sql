
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'team'],
        pre_hook='SELECT setseed(0.5)'
    )
}}

------------------------------------------------

WITH

country_stats AS (

    SELECT * FROM {{ ref('dbt_cricket', 'fct_country_stats') }}

),

weighted_stats AS (

    SELECT * FROM {{ ref('dbt_cricket', 'fct_weighted_stats') }}

),

double_stats AS (

    SELECT

        match_id,
        hash_id,
        start_date,
        team,
        opponent,
        home_advantage - LAG (home_advantage) OVER w AS home_advantage,
        chasing_advantage - LAG(chasing_advantage) OVER w AS chasing_advantage,
        team_strength - LAG (team_strength) OVER w AS delta_team_strength

    FROM weighted_stats

    WINDOW w AS (PARTITION BY match_id ORDER BY RANDOM())

)

SELECT * FROM double_stats

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}


