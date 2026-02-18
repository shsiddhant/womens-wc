WITH raw_json AS (
    SELECT * FROM
    {{ source('raw_data', 'raw_json') }}
)
SELECT
    DISTINCT ON (match_id)
    (data->'match_id')::INTEGER AS match_id,
    v.venue_id,
    TO_DATE(data->'info'->'dates'->> 0, 'YYYY-MM-DD') AS start_date,
    data->'info'->'event'->>'name' AS event_name,
    (data->'info'->'event'->>'match_number')::INTEGER AS event_match_number,
    data->'info'->'toss'->>'winner' AS toss_winner,
    data->'info'->'toss'->>'decision' AS toss_decision,
    CASE
        -- If 'winner' key is there, then put winner,
        WHEN data->'info'->'outcome'->>'winner' IS NOT NULL
            THEN 'winner'
        -- If no winner and 'result' is present, then put that ('tie' / 'no result'
        WHEN data->'info'->'outcome'->>'result' IS NOT NULL
            THEN data->'info'->'outcome'->>'result' END
    AS outcome_type,
    data->'info'->'outcome'->'by' AS outcome_margin,
    data->'info'->'outcome'->>'winner' AS winner,
    data->'info'->>'player_of_match' AS player_of_match
    FROM raw_json
    LEFT JOIN {{ ref('dbt_cricket', 'venues') }} AS v
    ON data->'info'->>'venue' = v.venue_name
