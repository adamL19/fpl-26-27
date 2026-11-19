-- FPL live dashboard schema (SQLite).  Prices are stored in tenths of a million.

CREATE TABLE IF NOT EXISTS players (
    id INTEGER PRIMARY KEY,
    code INTEGER,
    web_name TEXT,
    first_name TEXT,
    second_name TEXT,
    team_id INTEGER,
    position INTEGER,            -- 1 GK, 2 DEF, 3 MID, 4 FWD
    now_cost INTEGER,
    start_cost INTEGER,
    status TEXT,                 -- a available, d doubtful, i injured, s suspended, u unavailable, n not in squad
    chance_next REAL,            -- 0-100 or NULL
    chance_this REAL,
    news TEXT,
    minutes INTEGER,
    starts INTEGER,
    goals INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    saves INTEGER,
    bonus INTEGER,
    yellow INTEGER,
    red INTEGER,
    own_goals INTEGER,
    pens_saved INTEGER,
    pens_missed INTEGER,
    xg REAL,
    xa REAL,
    xgi REAL,
    xgc REAL,
    dc REAL,                     -- season total defensive-contribution count
    total_points INTEGER,
    selected_pct REAL,
    form REAL,
    ep_next REAL,                -- FPL's own expected points (used only as a sanity column)
    pen_order INTEGER,           -- 1 = first choice penalty taker
    fk_order INTEGER,            -- direct free kicks
    corner_order INTEGER,        -- corners & indirect free kicks
    fetched_at TEXT
);

CREATE TABLE IF NOT EXISTS teams (
    id INTEGER PRIMARY KEY,
    name TEXT,
    short_name TEXT,
    strength INTEGER,
    strength_attack_home INTEGER,
    strength_attack_away INTEGER,
    strength_defence_home INTEGER,
    strength_defence_away INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,      -- gameweek number
    name TEXT,
    deadline_time TEXT,
    finished INTEGER,
    is_current INTEGER,
    is_next INTEGER,
    data_checked INTEGER
);

CREATE TABLE IF NOT EXISTS fixtures (
    id INTEGER PRIMARY KEY,
    event INTEGER,               -- NULL while unscheduled / postponed
    team_h INTEGER,
    team_a INTEGER,
    kickoff_time TEXT,
    finished INTEGER,
    started INTEGER,
    team_h_score INTEGER,
    team_a_score INTEGER,
    team_h_difficulty INTEGER,
    team_a_difficulty INTEGER
);

CREATE TABLE IF NOT EXISTS gw_history (
    player_id INTEGER,
    fixture INTEGER,
    round INTEGER,
    opponent_team INTEGER,
    was_home INTEGER,
    kickoff_time TEXT,
    minutes INTEGER,
    starts INTEGER,
    goals INTEGER,
    assists INTEGER,
    clean_sheets INTEGER,
    goals_conceded INTEGER,
    saves INTEGER,
    bonus INTEGER,
    bps INTEGER,
    yellow INTEGER,
    red INTEGER,
    own_goals INTEGER,
    xg REAL,
    xa REAL,
    xgi REAL,
    xgc REAL,
    dc REAL,
    total_points INTEGER,
    value INTEGER,
    PRIMARY KEY (player_id, fixture)
);

CREATE TABLE IF NOT EXISTS past_seasons (
    player_id INTEGER,
    season_name TEXT,
    start_cost INTEGER,
    end_cost INTEGER,
    total_points INTEGER,
    minutes INTEGER,
    goals INTEGER,
    assists INTEGER,
    xg REAL,
    xa REAL,
    PRIMARY KEY (player_id, season_name)
);

CREATE INDEX IF NOT EXISTS idx_history_player ON gw_history(player_id, round);
CREATE INDEX IF NOT EXISTS idx_fixtures_event ON fixtures(event);
