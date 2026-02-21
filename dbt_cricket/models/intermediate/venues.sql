
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key='venue_id'
    )
}}

------------------------------------------------

WITH

staging AS (

    SELECT * FROM {{ ref('dbt_cricket', 'stg_json') }}

),

venues_with_hash_id AS (
    
    SELECT 
        DISTINCT ON (venue_name)
        info->>'venue' AS venue_name,
        hash_id,
        {{ dbt_utils.generate_surrogate_key(["info->>'venue'"]) }} AS venue_id,
        city,
        country,
        {{ dbt.current_timestamp() }} AS last_update

    FROM staging
)

SELECT * FROM venues_with_hash_id

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
