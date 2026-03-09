
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key='match_id'
    )
}}

------------------------------------------------

WITH

double_stats AS (

    SELECT * FROM {{ ref('dbt_cricket', 'fct_double_stats') }}

),

features AS (

    SELECT
        
        DISTINCT ON (match_id)

        *

    FROM double_stats

    WHERE double_stats IS NOT NULL

)

SELECT * FROM features

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}



