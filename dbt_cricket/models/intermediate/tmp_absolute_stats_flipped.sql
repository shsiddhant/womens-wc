
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


absolute_stats_flipped AS (

    SELECT

        matches.match_id,
        matches.hash_id,
        match_teams.team AS team,
        innings.team AS opponent,

        SUM(weights.weight * innings.runs_scored) OVER w AS weighted_runs_conceded,
        SUM(weights.weight * innings.legal_deliveries) OVER w AS weighted_deliveries_bowled,
        SUM(weights.weight * innings.wickets_lost) OVER w AS weighted_wickets_taken

    FROM matches
    
    JOIN match_teams
        ON matches.match_id = match_teams.match_id

    JOIN weights
        ON match_teams.match_id = weights.match_id
        AND match_teams.team = weights.team

    LEFT JOIN innings
        ON innings.match_id = match_teams.match_id
        AND innings.team = match_teams.opponent

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
        --AND matches.start_date >= '01-01-2018'

    WINDOW w AS (
        PARTITION BY match_teams.team
        ORDER BY matches.start_date
        ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW
    )

)

SELECT * FROM absolute_stats_flipped

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
