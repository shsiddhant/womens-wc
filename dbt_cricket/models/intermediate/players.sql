WITH raw_json AS (
    SELECT * FROM
    {{ source('raw_data', 'raw_json') }}
)
SELECT
    DISTINCT ON (player_id)
    data->'info'->'registry'->'people'->>pl.player_name AS player_id,
    pl.player_name,
    t.team_id AS team_id
    FROM raw_json
    CROSS JOIN LATERAL jsonb_each(data->'info'->'players') AS p(team, players)
    CROSS JOIN LATERAL jsonb_array_elements_text(p.players) AS pl(player_name)
    JOIN {{ ref('dbt_cricket', 'teams') }} t ON p.team = t.team
