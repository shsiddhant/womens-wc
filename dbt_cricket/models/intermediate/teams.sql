
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key='team_id'
    )
}}

------------------------------------------------

WITH

staging AS (
    SELECT * FROM {{ ref('dbt_cricket', 'stg_json') }}
),

teams AS (

    SELECT
        DISTINCT ON (team, format)
        {{ dbt_utils.generate_surrogate_key([
            't.team',
            "info->>'match_type'"
            ])
        }} AS team_id,
        staging.hash_id,
        t.team AS team,
        info->>'match_type' AS format,
        {{ dbt.current_timestamp() }} AS last_update
    
    FROM staging

    CROSS JOIN
        LATERAL jsonb_array_elements_text(info->'teams') AS t(team)
)

SELECT * FROM teams

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}
