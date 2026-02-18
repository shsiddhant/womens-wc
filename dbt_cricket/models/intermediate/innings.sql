SELECT
    match_id,
    innings_number,
    MAX(team) AS team,
    SUM(runs) AS runs_scored,
    COUNT(player_out) AS wickets_lost,
    COUNT(*) FILTER (WHERE is_legal) AS legal_deliveries,
    SUM(wides) AS wides,
    SUM(noballs) AS noballs,
    SUM(byes) AS byes,
    SUM(legbyes) AS legbyes,
    SUM(extras) AS extras
    FROM {{ ref('dbt_cricket', 'deliveries') }}
    GROUP BY (match_id, innings_number)
