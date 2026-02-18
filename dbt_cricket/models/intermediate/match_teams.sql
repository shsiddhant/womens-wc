
WITH raw_json AS (
    SELECT * FROM
    {{ source('raw_data', 'raw_json') }}
)
SELECT
    DISTINCT ON (match_id, team)
    {{ dbt_utils.generate_surrogate_key([
        'm.match_id',
        't.team'
        ])
    }} AS match_team_id,
    m.match_id AS match_id,
    data->'info'->>'match_type' AS format,
    t.team AS team,
    data->'info'->'teams'->>((2-t.position)::int) AS opponent,
    (v.country = t.team) AS is_home,
    CASE
        WHEN m.toss_winner = t.team
            THEN m.toss_decision = 'bat'
        ELSE
            m.toss_decision = 'field'
        END
    AS batted_first,
    (m.winner = t.team) AS won_match
    FROM raw_json
    JOIN {{ ref('dbt_cricket', 'venues') }} AS v
    ON (v.venue_name = data->'info'->>'venue')
    JOIN {{ ref('dbt_cricket', 'matches') }} AS m
    ON (data->'match_id')::INTEGER = m.match_id
    CROSS JOIN LATERAL jsonb_array_elements_text(data->'info'->'teams')
    WITH ORDINALITY AS t(team, position)
