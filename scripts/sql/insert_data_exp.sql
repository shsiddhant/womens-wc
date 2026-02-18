INSERT INTO venues (venue_name, city, country)
SELECT
    DISTINCT ON (venue_name)
    data->'info'->>'venue' AS venue_name,
    data->'info'->>'city' AS city,
    ctc.country AS country
    FROM json_table
    JOIN city_country AS ctc
    ON
    (ctc.city = data->'info'->>'city') OR
    (ctc.city IS NULL AND ctc.venue= data->'info'->>'venue')
    ON CONFLICT (venue_name) DO NOTHING;



INSERT INTO teams (team, format)
SELECT DISTINCT
t.team,
data->'info'->>'match_type'
FROM json_table
CROSS JOIN LATERAL jsonb_array_elements_text(data->'info'->'teams') AS t(team)
ON CONFLICT (team, format) DO NOTHING;


INSERT INTO players (player_id, player_name, team_id)
SELECT DISTINCT ON
(player_id)
data->'info'->'registry'->'people'->>pl.player_name AS player_id,
pl.player_name,
t.team_id AS team_id
FROM
    json_table
    CROSS JOIN LATERAL jsonb_each(data->'info'->'players') AS p(team, players)
    CROSS JOIN LATERAL jsonb_array_elements_text(p.players) AS pl(player_name)
    JOIN teams t ON p.team = t.team
ON CONFLICT (player_id) DO NOTHING;

INSERT INTO matches
(match_id, venue_id, start_date, event_name, event_match_number, toss_winner, toss_decision, outcome_type, outcome_margin, winner, player_of_the_match)
SELECT DISTINCT ON (match_id)
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
LEFT JOIN venues v
ON data->'info'->>'venue' = v.venue_name
ON CONFLICT (match_id) DO NOTHING;


INSERT INTO match_teams (match_id, format, team, opp_name, is_home, batted_first, won_match)
SELECT DISTINCT ON (match_id, team)
m.match_id,
jt.data->'info'->>'match_type',
t.team,
data->'info'->'teams'->>((2-t.position)::int),
(v.country = t.team),
CASE
    WHEN m.toss_winner = t.team
        THEN m.toss_decision = 'bat'
    ELSE
        m.toss_decision = 'field'
    END
AS batted_first,
(m.winner = t.team) AS won_match
FROM
json_table jt
JOIN venues v ON (v.venue_name = jt.data->'info'->>'venue')
JOIN matches m ON (jt.data->'match_id')::INTEGER = m.match_id
CROSS JOIN LATERAL jsonb_array_elements_text(jt.data->'info'->'teams') WITH ORDINALITY AS t(team, position)
ON CONFLICT (match_id, team) DO NOTHING;

EXPLAIN ANALYZE
INSERT INTO deliveries (
match_id,
innings_number,
team,
--opp_name,
over_number,
ball_in_over,
runs,
wides,
noballs,
byes,
legbyes,
extras,
batter,
bowler,
is_legal,
player_out,
dismissal_mode)

SELECT
st.match_id AS match_id,
st.n_innings AS innings_number,
st.team AS team,
st.n_over AS over_number,
st.n_delivery AS ball_in_over,
COALESCE ((st.delivery->'runs'->'total')::integer, 0),
COALESCE ((st.delivery->'extras'->'wides')::integer, 0),
COALESCE ((st.delivery->'extras'->'noballs')::integer, 0),
COALESCE ((st.delivery->'extras'->'byes')::integer, 0),
COALESCE ((st.delivery->'extras'->'legbyes')::integer, 0),
COALESCE ((st.delivery->'runs'->'extras')::integer, 0),
st.delivery->>'batter',
st.delivery->>'bowler',
((st.delivery->'extras'->'wides') IS NULL) AND
((st.delivery->'extras'->'noballs') IS NULL),
st.delivery->'wickets'->0->>'player_out',
st.delivery->'wickets'->0->>'kind'
FROM stage_deliveries st
ON CONFLICT DO NOTHING;

INSERT INTO innings(
    match_id,
    innings_number,
    team,
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
MAX(team) AS team,
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
GROUP BY (match_id, innings_number)
ON CONFLICT (match_id, innings_number) DO NOTHING;
