
-- Configuration

{{
    config(
        materialized='incremental',
        unique_key=['match_id', 'player_out']
    )
}}

------------------------------------------------

WITH

deliveries AS (

	SELECT * FROM {{ ref('dbt_cricket', 'deliveries') }}

),

dismissed_players AS (

	SELECT
		DISTINCT ON (match_id, player_out)
			
			match_id,
			player_out,
			hash_id

	FROM deliveries

	WHERE player_out IS NOT NULL

)


SELECT * FROM dismissed_players

{% if is_incremental() %}

    WHERE hash_id NOT IN (SELECT DISTINCT hash_id FROM {{ this }})

{% endif %}

