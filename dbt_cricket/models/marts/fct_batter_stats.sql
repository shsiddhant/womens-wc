
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'player_name']
    )
}}

------------------------------------------------

WITH

match_players AS (

    SELECT * FROM {{ ref('dbt_cricket', 'match_players') }}

),

match_teams AS (

    SELECT * FROM {{ ref('dbt_cricket', 'match_teams') }}

),

dismissed_players AS (

	SELECT * FROM {{ ref('dbt_cricket', 'fct_dismissed') }}

),

deliveries AS (

    SELECT * FROM {{ ref('dbt_cricket', 'deliveries') }}

),

batter_stats_without_dismissal AS (

    SELECT

        match_players.match_id,
        match_players.player_name,

        MAX(match_players.hash_id) AS hash_id,
        MAX(match_players.team) AS team,
        MAX(match_teams.opponent) AS opponent,

        COALESCE(

            SUM(deliveries.batter_runs), 0

        ) AS runs,

        SUM(
            CASE
                WHEN deliveries.wides = 0 THEN 1
               
                ELSE 0

            END

        ) AS balls

    FROM match_players

    JOIN match_teams
        ON match_players.match_id = match_teams.match_id
        AND match_players.team = match_teams.team

    LEFT JOIN deliveries
        ON match_players.match_id = deliveries.match_id
        AND match_players.player_name = deliveries.batter

    GROUP BY
        match_players.match_id,
        match_players.player_name
),

batter_stats AS (

		SELECT

				batter_stats_without_dismissal.*,

    		(dismissed_players.player_out IS NOT NULL) AS is_dismissed

		FROM batter_stats_without_dismissal

		LEFT JOIN dismissed_players
				ON batter_stats_without_dismissal.match_id = dismissed_players.match_id
        AND batter_stats_without_dismissal.player_name = dismissed_players.player_out

)

SELECT * FROM batter_stats

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
