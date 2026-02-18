WITH raw_json AS (
    SELECT * FROM
    {{ source('raw_data', 'raw_json') }}
)
SELECT
    DISTINCT ON (team, format)
    {{ dbt_utils.generate_surrogate_key([
        't.team',
        "data->'info'->>'match_type'"
        ])
    }} AS team_id,
    t.team AS team,
    data->'info'->>'match_type' AS format,
    {{ dbt.current_timestamp() }} AS last_update
    FROM raw_json
    CROSS JOIN LATERAL jsonb_array_elements_text(data->'info'->'teams') AS t(team)
