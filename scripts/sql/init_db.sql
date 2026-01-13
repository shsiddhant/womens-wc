DROP TABLE IF EXISTS
batting_innings,
bowling_innings,
innings,
deliveries,
match_teams,
matches,
players,
teams,
venues,
city_country,
json_table;

CREATE TABLE json_table (
    id SERIAL,
    data JSONB,
    PRIMARY KEY (id)
);

-- venue name column is only used if city name is missing in match JSON
CREATE TABLE city_country (
    id SERIAL,
    city TEXT,
    country TEXT,
    venue TEXT,
    UNIQUE (city, venue),
    PRIMARY KEY (id)
);


CREATE TABLE venues (
    venue_id SERIAL,
    venue_name TEXT NOT NULL,
    city TEXT,
    country TEXT NOT NULL,
    last_update TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP NOT NULL,
		PRIMARY KEY (venue_id)
);

CREATE TABLE teams (
    team_id SERIAL PRIMARY KEY,
    team_name TEXT NOT NULL,
    format TEXT NOT NULL,
    UNIQUE (team_name, format)
);

CREATE TABLE players (
    player_id TEXT,
    player_name TEXT NOT NULL,
    team_id INTEGER,
    PRIMARY KEY (player_id),
    CONSTRAINT players_team_id_fk FOREIGN KEY (team_id) REFERENCES teams(team_id)
);

CREATE TABLE matches (
    match_id INTEGER NOT NULL,
    venue_id INTEGER,
    start_date DATE NOT NULL,
    event_name TEXT,
    event_match_number INTEGER,
    toss_winner TEXT,
    toss_decision TEXT,
    outcome_type TEXT NOT NULL,
    outcome_margin JSONB,
    winner TEXT,
    player_of_the_match TEXT,
    PRIMARY KEY (match_id),
    CONSTRAINT matches_venue_id_fk FOREIGN KEY (venue_id) REFERENCES venues(venue_id)
);

CREATE TABLE match_teams (
    match_team_id SERIAL,
    match_id INTEGER,
    format TEXT NOT NULL,
    team_name TEXT NOT NULL,
    opp_name TEXT NOT NULL,
    is_home boolean,
    batted_first boolean,
    won_match boolean,
    PRIMARY KEY (match_team_id),
    CONSTRAINT match_teams_match_id_fk FOREIGN KEY (match_id) REFERENCES matches(match_id)
);

CREATE TABLE deliveries (
    match_id INTEGER NOT NULL,
    innings_number INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    opp_name TEXT, -- NOT NULL,
    is_legal BOOLEAN,
    runs INTEGER,
    wides INTEGER,
    noballs INTEGER,
    byes INTEGER,
    legbyes INTEGER,
    extras INTEGER,
    batter_id TEXT NOT NULL,
    bowler_id TEXT NOT NULL,
    player_out TEXT,
    dismissal_mode TEXT,
    CONSTRAINT deliveries_match_fk
        FOREIGN KEY (match_id)
        REFERENCES matches(match_id),
    CONSTRAINT deliveries_batter_fk
        FOREIGN KEY (batter_id)
        REFERENCES players(player_id),
    CONSTRAINT deliveries_bowler_fk
        FOREIGN KEY (bowler_id)
        REFERENCES players(player_id)
);

CREATE TABLE innings (
    innings_id SERIAL,
    match_id INTEGER,
    innings_number INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    opp_name TEXT, -- NOT NULL,
    runs_scored INTEGER,
    wickets_lost INTEGER,
    legal_deliveries INTEGER,
    wides INTEGER,
    noballs INTEGER,
    byes INTEGER,
    legbyes INTEGER,
    extras INTEGER,
    PRIMARY KEY (innings_id),
    UNIQUE (match_id, innings_number),
    FOREIGN KEY (match_id) REFERENCES matches (match_id)
);

CREATE TABLE batting_innings (
    match_id INTEGER NOT NULL,
    innings_number INTEGER NOT NULL,
    batter_id TEXT,
    runs_scored INTEGER,
    deliveries_played INTEGER,
    dismissed boolean NOT NULL,
    dismissal_type TEXT,
    bowler_id TEXT,
    PRIMARY KEY (match_id, innings_number),
    FOREIGN KEY (batter_id) REFERENCES players(player_id),
    FOREIGN KEY (bowler_id) REFERENCES players(player_id),
    FOREIGN KEY (match_id, innings_number) REFERENCES innings(match_id, innings_number)
);

CREATE TABLE bowling_innings (
    match_id INTEGER NOT NULL,
    innings_number INTEGER NOT NULL,
    bowler_id TEXT,
    deliveries_bowled INTEGER,
    runs_conceded INTEGER,
    wickets_taken INTEGER,
    PRIMARY KEY (match_id, innings_number),
    FOREIGN KEY (bowler_id) REFERENCES players(player_id),
    FOREIGN KEY (match_id, innings_number) REFERENCES innings(match_id, innings_number)
);
