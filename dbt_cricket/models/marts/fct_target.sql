
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key='match_id'
    )
}}

------------------------------------------------

WITH

match_teams AS (

    SELECT * FROM {{ ref('dbt_cricket', 'match_teams') }}

),

features AS (

    SELECT * FROM {{ ref('dbt_cricket', 'fct_features') }}

),

target AS (

    SELECT

        DISTINCT ON (match_id)

        match_id,
        features.hash_id,
        won_match AS result

    FROM match_teams

    JOIN features
       USING (match_id, team)

)

SELECT * FROM target

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}

