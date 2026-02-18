WITH raw_json AS (
    SELECT * FROM
    {{ source('raw_data', 'raw_json') }}
),
ctc AS (
    SELECT * FROM
    {{ source('raw_data', 'city_country') }}
)
SELECT
    DISTINCT ON (venue_name)
    data->'info'->>'venue' AS venue_name,
    {{ dbt_utils.generate_surrogate_key(["data->'info'->>'venue'"]) }} AS venue_id,
    data->'info'->>'city' AS city,
    ctc.country AS country,
    {{ dbt.current_timestamp() }} AS last_update
    FROM raw_json
    JOIN ctc
    ON
    (ctc.city = data->'info'->>'city') OR
    (ctc.city IS NULL AND ctc.venue= data->'info'->>'venue')
