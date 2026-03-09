
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'team']
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

absolute_stats AS (

    SELECT * FROM {{ ref('dbt_cricket', 'tmp_absolute_stats') }}

),

absolute_stats_flipped AS (

    SELECT * FROM {{ ref('dbt_cricket', 'tmp_absolute_stats_flipped') }}

),

weighted_stats AS (

    SELECT

        matches.match_id,
        matches.hash_id,
        matches.start_date,
        absolute_stats.team,
        absolute_stats.opponent,

        absolute_stats.home_advantage,
        absolute_stats.chasing_advantage,

        ROUND((100 * absolute_stats.weighted_wins /
                absolute_stats.weighted_matches_count)::numeric, 2)
        AS weighted_win_percentage,
        
        ROUND((absolute_stats.weighted_runs_scored /
                absolute_stats.weighted_wickets_lost)::numeric, 2)
        AS weighted_batting_average,

        ROUND((absolute_stats_flipped.weighted_runs_conceded /
                absolute_stats_flipped.weighted_wickets_taken)::numeric, 2)
        AS weighted_bowling_average,

        ROUND((6 * absolute_stats_flipped.weighted_runs_conceded /
                absolute_stats_flipped.weighted_deliveries_bowled)::numeric, 2)
        AS weighted_bowling_economy
    
    FROM absolute_stats

    JOIN absolute_stats_flipped
        ON absolute_stats.match_id = absolute_stats_flipped.match_id
        AND absolute_stats.team = absolute_stats_flipped.team

    LEFT JOIN match_teams
        ON match_teams.match_id = absolute_stats.match_id
        AND match_teams.team = absolute_stats.team

    JOIN matches
        ON matches.match_id = match_teams.match_id

)

SELECT

    *,
    0.01 * weighted_win_percentage * weighted_batting_average / weighted_bowling_average AS team_strength

FROM weighted_stats

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
