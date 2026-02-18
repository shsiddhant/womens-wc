BEGIN;
DROP TABLE IF EXISTS weights;
DROP TABLE IF EXISTS wtst;
DROP TABLE IF EXISTS features;
DROP TABLE IF EXISTS target;

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

CREATE TABLE target(
    match_id INT NOT NULL,
    result INT NOT NULL
);
    
    

CREATE TABLE weights (
match_id INT NOT NULL,
start_date DATE NOT NULL,
team TEXT NOT NULL,
n_matches INT,
weight REAL,
CONSTRAINT wt_gt_zero CHECK (weight > 0),
CONSTRAINT wt_match_id_fk FOREIGN KEY(match_id) REFERENCES matches(match_id)
);


WITH ct AS (
SELECT mt.match_id,
m.start_date,
mt.team,
COUNT(*) OVER w AS n_matches
FROM
matches m
JOIN match_teams mt
ON m.match_id = mt.match_id
window w AS (PARTITION BY mt.team ORDER BY m.start_date ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW)
)
INSERT INTO weights (
match_id,
start_date,
team,
n_matches,
weight
)
SELECT
ct.match_id,
ct.start_date,
ct.team,
ct.n_matches,
EXP(- LN(2) * n_matches / 20) AS weight
FROM ct;

CREATE TEMPORARY TABLE absolute_stats AS
SELECT
i.match_id AS match_id,
i.team AS team,
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
RIGHT JOIN match_teams mt
ON (i.match_id = mt.match_id) AND (i.team = mt.team)
JOIN weights wt ON wt.match_id = mt.match_id AND wt.team = mt.team
RIGHT JOIN matches m ON mt.match_id = m.match_id
WHERE
CASE
    WHEN NOT %s THEN (m.outcome_type = 'winner')
    ELSE True
    END
AND
i.team IN %s AND i.opp_name IN %s AND m.start_date >= %s
window w AS (PARTITION BY i.team ORDER BY m.start_date ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW);

CREATE TEMPORARY TABLE absolute_stats_flipped AS
SELECT
i.match_id AS match_id,
mt.team AS team,
mt.opp_name AS opponent,
SUM(wt.weight * i.runs_scored) OVER w AS wt_runs_conceded,
SUM(wt.weight * i.legal_deliveries) OVER w AS wt_deliveries_bowled,
SUM(wt.weight * i.wickets_lost) OVER w AS wt_wickets_taken
FROM innings i
RIGHT JOIN match_teams mt
ON (i.match_id = mt.match_id) AND (i.opp_name = mt.team)
JOIN weights wt ON wt.match_id = mt.match_id AND wt.team = mt.team
RIGHT JOIN matches m ON mt.match_id = m.match_id
WHERE
CASE
    WHEN NOT %s THEN m.outcome_type = 'winner'
    ELSE True
    END
AND
mt.team IN %s AND mt.opp_name IN %s AND m.start_date >= %s
window w AS (PARTITION BY mt.team ORDER BY m.start_date ROWS UNBOUNDED PRECEDING EXCLUDE CURRENT ROW);

CREATE TABLE wtst AS (
SELECT
m.match_id AS match_id,
m.start_date AS start_date,
ast.team AS team,
ast.opponent AS opponent,
ast.home_advantage AS home_advantage,
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
AND mt.team = ast.team
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
home_advantage - LAG(home_advantage) OVER w AS  home_advantage,
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

INSERT INTO target (
    match_id,
    result
)
SELECT
    DISTINCT ON (ft.start_date, ft.match_id)
    ft.match_id,
    CASE
        WHEN mt.won_match IS NULL
            THEN NULL
        WHEN mt.won_match
            THEN 1
        WHEN mt.won_match IS NOT NULL AND (NOT mt.won_match)
            THEN 0
        END
    AS result
    FROM match_teams mt
    JOIN features ft
    ON mt.match_id = ft.match_id
    AND mt.team = ft.team;
    
COMMIT;
