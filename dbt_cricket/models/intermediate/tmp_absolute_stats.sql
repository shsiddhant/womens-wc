
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'team']
    )
}}

------------------------------------------------

{%
    set teams_list = [
        "Australia",
        "Bangladesh",
        "England",
        "India",
        "New Zealand",
        "Pakistan",
        "South Africa",
        "Sri Lanka"
    ]

%}

WITH

matches AS (

    SELECT * FROM {{ ref('dbt_cricket', 'matches') }}

),

match_teams AS (

    SELECT * FROM {{ ref('dbt_cricket', 'match_teams') }}

),

innings AS (

    SELECT * FROM {{ ref('dbt_cricket', 'innings') }}

),

weights AS (

    SELECT * FROM {{ ref('dbt_cricket', 'weights_by_team_window') }}

),

country_stats AS (

    SELECT * FROM {{ ref('dbt_cricket', 'fct_country_stats') }}

),

absolute_stats AS (

    SELECT

        matches.match_id,
        matches.hash_id,
        match_teams.team,
        match_teams.opponent,

        match_teams.batted_first::int * country_stats.chasing_advantage AS chasing_advantage,

        CASE
            WHEN match_teams.is_home THEN 1
            ELSE 0
            END
        AS home_advantage,

        SUM(
            weights.weight * (
                CASE
                    WHEN match_teams.won_match THEN 1
                    ELSE 0
                    END
                )
            ) OVER w AS weighted_wins,

        SUM(weights.weight) OVER w AS weighted_matches_count,
        SUM(weights.weight * innings.runs_scored) OVER w AS weighted_runs_scored,
        SUM(weights.weight * innings.legal_deliveries) OVER w AS weighted_deliveries_faced,
        SUM(weights.weight * innings.wickets_lost) OVER w AS weighted_wickets_lost

    FROM matches

    JOIN country_stats
        ON matches.match_id = country_stats.match_id
    
    JOIN match_teams
        ON matches.match_id = match_teams.match_id

    JOIN weights
        ON match_teams.match_id = weights.match_id
        AND match_teams.team = weights.team

    LEFT JOIN innings
        ON innings.match_id = match_teams.match_id
        AND innings.team = match_teams.team

    WHERE
        innings.team IN (
            {% for team in teams_list %}
            '{{ team }}'
            {% if not loop.last %}, {% endif %}
            {% endfor %}
        )
        AND match_teams.opponent IN (
            {% for team in teams_list %}
            '{{ team }}'
            {% if not loop.last %}, {% endif %}
            {% endfor %}
        )
        AND innings.innings_number IN (0, 1)
--        AND matches.start_date >= '01-01-2018'

    WINDOW w AS (
        PARTITION BY innings.team
        ORDER BY matches.start_date
        ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW
    )

)

SELECT * FROM absolute_stats

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
