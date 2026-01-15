BEGIN;
DROP TABLE IF EXISTS weights;
DROP TABLE IF EXISTS wtst;
DROP TABLE IF EXISTS features;

CREATE TABLE features (
match_id INT NOT NULL,
start_date DATE NOT NULL,
team TEXT NOT NULL,
opponent TEXT NOT NULL,
home_advantage INT,
delta_wt_win_perc REAL,
delta_wt_bat_avg REAL,
delta_wt_bat_sr REAL,
delta_wt_bowl_avg REAL,
delta_wt_bowl_econ REAL
);

CREATE TABLE weights (
match_id INT NOT NULL,
start_date DATE NOT NULL,
days INT,
weight REAL,
CONSTRAINT wt_gt_zero CHECK (weight > 0)
);

WITH d(match_id, start_date, days) AS (
SELECT
m.match_id,
m.start_date,
m.start_date - MIN(m.start_date) OVER()
FROM matches m
),
med(value) AS (
SELECT
percentile_disc(0.5) WITHIN GROUP (ORDER BY d.days)
FROM d
)
INSERT INTO weights (
match_id,
start_date,
days,
weight
)
SELECT
d.match_id,
d.start_date,
d.days - med.value,
EXP(0.0038 * (d.days-med.value)) AS weight
FROM d
CROSS JOIN med;

CREATE TEMPORARY TABLE absolute_stats AS
SELECT
i.match_id AS match_id,
i.team_name AS team,
i.opp_name AS opponent,
CASE
    WHEN mt.is_home THEN 1
    ELSE 0
    END
AS home_advantage,
SUM(wt.weight * (CASE WHEN mt.won_match THEN 1 ELSE 0 END)) OVER w AS wt_wins,
SUM(wt.weight) OVER w AS wt_matches,
SUM(wt.weight * i.runs_scored) OVER w AS wt_runs_scored,
SUM(wt.weight * i.legal_deliveries) OVER w AS wt_deliveries_faced,
SUM(wt.weight * i.wickets_lost) OVER w AS wt_wickets_lost
FROM innings i
JOIN match_teams mt
ON (i.match_id = mt.match_id) AND (i.team_name = mt.team_name)
JOIN matches m ON mt.match_id = m.match_id
JOIN weights wt ON wt.match_id = m.match_id
WHERE
i.team_name IN %s AND i.opp_name IN %s AND m.start_date >= %s
window w AS (PARTITION BY i.team_name ORDER BY m.start_date ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW);

CREATE TEMPORARY TABLE absolute_stats_flipped AS
SELECT
i.match_id AS match_id,
mt.team_name AS team,
mt.opp_name AS opponent,
CASE WHEN mt.is_home THEN 0 ELSE -1 END AS away_disadvantage,
SUM(wt.weight * i.runs_scored) OVER w AS wt_runs_conceded,
SUM(wt.weight * i.legal_deliveries) OVER w AS wt_deliveries_bowled,
SUM(wt.weight * i.wickets_lost) OVER w AS wt_wickets_taken
FROM innings i
JOIN match_teams mt
ON (i.match_id = mt.match_id) AND (i.opp_name = mt.team_name)
JOIN matches m ON mt.match_id = m.match_id
JOIN weights wt ON wt.match_id = m.match_id
WHERE
i.team_name IN %s AND i.opp_name IN %s AND m.start_date >= %s
window w AS (PARTITION BY mt.team_name ORDER BY m.start_date ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW);

CREATE TABLE wtst AS (
SELECT
m.match_id AS match_id,
m.start_date AS start_date,
ast.team AS team,
ast.opponent AS opponent,
ast.home_advantage + asf.away_disadvantage AS home_advantage,
ROUND((100* wt_wins / wt_matches)::numeric, 2) AS wt_win_perc,
ROUND((wt_runs_scored / wt_wickets_lost)::numeric, 2) AS wt_bat_avg,
ROUND((100 * wt_runs_scored / wt_deliveries_faced)::numeric, 2) AS wt_bat_sr,
ROUND((wt_runs_conceded / wt_wickets_taken)::numeric, 2) AS wt_bowl_avg,
ROUND((6 * wt_runs_conceded / wt_deliveries_bowled)::numeric, 2) AS wt_bowl_econ
FROM absolute_stats ast
JOIN absolute_stats_flipped asf
ON asf.match_id = ast.match_id
AND asf.team = ast.team
JOIN match_teams mt
ON mt.match_id  = ast.match_id
AND mt.team_name = ast.team
JOIN matches m
ON m.match_id = mt.match_id
);

SELECT setseed(0.5);

CREATE TEMPORARY TABLE double_stats AS (
SELECT
match_id AS match_id,
start_date AS start_date,
team AS team,
opponent AS opponent,
home_advantage AS home_advantage,
wt_win_perc - LAG (wt_win_perc) OVER w AS delta_wt_win_perc,
wt_bat_avg - LAG (wt_bat_avg) OVER w AS delta_wt_bat_avg,
wt_bat_sr - LAG (wt_bat_sr) OVER w AS delta_wt_bat_sr,
wt_bowl_avg - LAG (wt_bowl_avg) OVER w AS delta_wt_bowl_avg,
wt_bowl_econ - LAG (wt_bowl_econ) OVER w AS delta_wt_bowl_econ
FROM
wtst
window w AS (PARTITION BY match_id ORDER BY RANDOM())
);
INSERT INTO features (
match_id,
start_date,
team,
opponent,
home_advantage,
delta_wt_win_perc,
delta_wt_bat_avg,
delta_wt_bat_sr,
delta_wt_bowl_avg,
delta_wt_bowl_econ
)
SELECT
DISTINCT ON (match_id)
match_id,
start_date,
team,
opponent,
home_advantage,
delta_wt_win_perc,
delta_wt_bat_avg,
delta_wt_bat_sr,
delta_wt_bowl_avg,
delta_wt_bowl_econ
FROM double_stats
WHERE delta_wt_win_perc IS NOT NULL
ORDER BY match_id, start_date;
COMMIT;
