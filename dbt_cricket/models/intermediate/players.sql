-- Configuration

{{
    config(
        materialized='incremental',
        unique_key='player_id'
    )
}}

------------------------------------------------
WITH

staging AS (
    
    SELECT * FROM {{ ref('dbt_cricket', 'stg_json') }}

),

teams AS (

    SELECT * FROM {{ ref('dbt_cricket', 'teams') }}

),

players AS (
    
    SELECT
        DISTINCT ON (player_id)
        staging.info->'registry'->'people'->>match_players_names.player_name AS player_id,
        staging.hash_id,
        match_players_names.player_name,
        teams.team_id AS team_id

    FROM staging 
    
    CROSS JOIN
        LATERAL jsonb_each(info->'players') AS match_players(team, players)
    
    CROSS JOIN
        LATERAL jsonb_array_elements_text(match_players.players) AS match_players_names(player_name)
    
    JOIN teams
        ON match_players.team = teams.team

)

SELECT * FROM players

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
