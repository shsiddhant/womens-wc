INSERT INTO venues (venue_name, city, country)
SELECT
data->'info'->>'venue' AS venue_name,
MAX(data->'info'->>'city') AS city,
MAX(ctc.country) AS country
FROM json_table
JOIN city_country AS ctc
ON (ctc.city = data->'info'->>'city') OR (ctc.city IS NULL AND ctc.venue= data->'info'->>'venue')
GROUP BY venue_name;

INSERT INTO teams (team_name, format)
SELECT
t.team_name,
data->'info'->>'match_type'
FROM json_table
CROSS JOIN LATERAL jsonb_array_elements_text(data->'info'->'teams') AS t(team_name)
ON CONFLICT (team_name, format) DO NOTHING;

INSERT INTO players (player_id, player_name, team_id)
SELECT DISTINCT ON
(player_id)
data->'info'->'registry'->'people'->>pl.player_name AS player_id,
pl.player_name,
t.team_id AS team_id
FROM json_table
CROSS JOIN LATERAL jsonb_each(data->'info'->'players') AS p(team_name, players)
CROSS JOIN LATERAL jsonb_array_elements_text(p.players) AS pl(player_name)
JOIN teams t ON p.team_name = t.team_name;

INSERT INTO matches
(match_id, venue_id, start_date, event_name, event_match_number, toss_winner, toss_decision, outcome_type, outcome_margin, winner, player_of_the_match)
SELECT
(data->'match_id')::INTEGER AS match_id,
venue_id,
TO_DATE(data->'info'->'dates'->> 0, 'YYYY-MM-DD') AS start_date, -- Cast TEXT to DATE
data->'info'->'event'->>'name',
(data->'info'->'event'->>'match_number')::INTEGER, -- CAST TEXT to INTEGER
data->'info'->'toss'->>'winner',
data->'info'->'toss'->>'decision',
CASE
    -- If 'winner' key is there, then put winner,
    WHEN data->'info'->'outcome'->>'winner' IS NOT NULL
        THEN 'winner'
    -- If no winner and 'result' is present, then put that ('tie' / 'no result'
    WHEN data->'info'->'outcome'->>'result' IS NOT NULL
        THEN data->'info'->'outcome'->>'result' END,
data->'info'->'outcome'->'by',
data->'info'->'outcome'->>'winner',
data->'info'->>'player_of_match'
FROM json_table
JOIN venues v
ON data->'info'->>'venue' = v.venue_name;

INSERT INTO match_teams (match_id, format, team_name, opp_name, is_home, batted_first, won_match)
SELECT
m.match_id,
jt.data->'info'->>'match_type',
t.team_name,
data->'info'->'teams'->>((2-t.position)::int),
(v.country = t.team_name),
CASE
    WHEN m.toss_winner = t.team_name
        THEN m.toss_decision = 'bat'
    ELSE
        m.toss_decision = 'field'
    END
AS batted_first,
(m.winner = t.team_name) AS won_match
FROM
json_table jt
JOIN venues v ON (v.venue_name = jt.data->'info'->>'venue')
JOIN matches m ON (jt.data->'match_id')::INTEGER = m.match_id
CROSS JOIN LATERAL jsonb_array_elements_text(jt.data->'info'->'teams') WITH ORDINALITY AS t(team_name, position);

INSERT INTO deliveries (
match_id,
innings_number,
team_name,
opp_name,
runs,
wides,
noballs,
byes,
legbyes,
extras,
batter_id,
bowler_id,
is_legal,
player_out,
dismissal_mode)

SELECT
mt.match_id AS match_id,
i.position AS innings_number,
mt.team_name AS team_name,
mt.opp_name AS opp_name,
COALESCE ((d.delivery->'runs'->'total')::integer, 0),
COALESCE ((d.delivery->'extras'->'wides')::integer, 0),
COALESCE ((d.delivery->'extras'->'noballs')::integer, 0),
COALESCE ((d.delivery->'extras'->'byes')::integer, 0),
COALESCE ((d.delivery->'extras'->'legbyes')::integer, 0),
COALESCE ((d.delivery->'runs'->'extras')::integer, 0),
rb.value AS batter_id,
rw.value AS bowler_id,
((d.delivery->'extras'->'wides') IS NULL) AND
((d.delivery->'extras'->'noballs') IS NULL),
d.delivery->'wickets'->0->>'player_out',
d.delivery->'wickets'->0->>'kind'

FROM json_table

CROSS JOIN LATERAL jsonb_array_elements(data->'innings')
  WITH ORDINALITY AS i(innings, position)
JOIN match_teams mt
  ON mt.match_id = (data->'match_id')::integer AND i.innings->>'team' = mt.team_name
CROSS JOIN LATERAL jsonb_array_elements(i.innings->'overs') ov
CROSS JOIN LATERAL jsonb_array_elements(ov->'deliveries') d(delivery)
JOIN jsonb_each_text(data->'info'->'registry'->'people') rb
  ON rb.key = d.delivery->>'batter'

JOIN jsonb_each_text(data->'info'->'registry'->'people') rw
  ON rw.key = d.delivery->>'bowler';

INSERT INTO innings(
    match_id,
    innings_number,
    team_name,
    opp_name,
    runs_scored,
    wickets_lost,
    legal_deliveries,
    wides,
    noballs,
    byes,
    legbyes,
    extras
)
SELECT
match_id,
innings_number,
MAX(team_name) AS team_name,
MAX(opp_name) AS opp_name,
SUM(runs) AS runs_scored,
COUNT(player_out) AS wickets_lost,
COUNT(*) FILTER (WHERE is_legal) AS legal_deliveries,
SUM(wides) AS wides,
SUM(noballs) AS noballs,
SUM(byes) AS byes,
SUM(legbyes) AS legbyes,
SUM(extras) AS extras

FROM deliveries
GROUP BY (match_id, innings_number);
