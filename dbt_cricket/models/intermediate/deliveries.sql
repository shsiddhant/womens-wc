WITH deliveries_json AS (
    SELECT * FROM
    {{ source('raw_data', 'deliveries_json') }}
)
SELECT
    dj.match_id AS match_id,
    dj.n_innings AS innings_number,
    dj.team AS team,
    dj.n_over AS over_number,
    dj.n_delivery AS ball_in_over,
    COALESCE ((dj.delivery->'runs'->'total')::integer, 0) AS runs,
    COALESCE ((dj.delivery->'extras'->'wides')::integer, 0) AS wides,
    COALESCE ((dj.delivery->'extras'->'noballs')::integer, 0) AS noballs,
    COALESCE ((dj.delivery->'extras'->'byes')::integer, 0) AS byes,
    COALESCE ((dj.delivery->'extras'->'legbyes')::integer, 0) AS legbyes,
    COALESCE ((dj.delivery->'runs'->'extras')::integer, 0) AS extras,
    dj.delivery->>'batter' AS batter,
    dj.delivery->>'bowler' AS bowler,
    ((dj.delivery->'extras'->'wides') IS NULL) AND
    ((dj.delivery->'extras'->'noballs') IS NULL)
    AS is_legal,
    dj.delivery->'wickets'->0->>'player_out' AS player_out,
    dj.delivery->'wickets'->0->>'kind' AS dismissal_mode
    FROM deliveries_json dj
