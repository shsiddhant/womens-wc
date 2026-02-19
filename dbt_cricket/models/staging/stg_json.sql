-- Configuration

{{
    config(
        materialized='incremental'
    )
}}

----------------------------------------------------

WITH 

raw_json AS (
    
    SELECT * FROM {{ source('raw_data', 'raw_json') }}

),

ctc AS (

    SELECT * FROM {{ source('raw_data', 'city_country') }}

),

staging AS (
    
    SELECT
        DISTINCT ON (match_id)

        (data->'match_id')::INTEGER AS match_id,
        {{ dbt_utils.generate_surrogate_key(["data"]) }} AS hash_id,
        data->'info' AS info,
        ctc.country AS country,
        {{ dbt.current_timestamp() }} AS last_update
    
    FROM raw_json
    
    LEFT JOIN ctc
        ON
        (ctc.city = data->'info'->>'city') OR
        (ctc.city IS NULL AND ctc.venue= data->'info'->>'venue')

)

SELECT * FROM staging

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT hash_id FROM {{ this }})

{% endif %}
