"""Central place for FPL rules and model tuning knobs.

Rules were checked against the official Premier League announcements for
2026/27 (two sets of chips, GW19 deadline for set one, five banked free
transfers, defensive contribution points continuing).  Anything marked
ASSUMPTION is a modelling choice you can tune.
"""
from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------
# Squad rules
# --------------------------------------------------------------------------
POS_NAME = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
SQUAD_QUOTA = {1: 2, 2: 5, 3: 5, 4: 3}
XI_MIN = {1: 1, 2: 3, 3: 2, 4: 1}
XI_MAX = {1: 1, 2: 5, 3: 5, 4: 3}
MAX_PER_CLUB = 3
BUDGET_TENTHS = 1000          # prices are stored in tenths of a million (100.0m)
HIT_COST = 4
MAX_BANKED_FT = 5

# --------------------------------------------------------------------------
# Chips: two sets of four.  Set 1 must be used before the GW19 deadline
# (Sat 2 Jan 2027), set 2 is available from GW20.
# --------------------------------------------------------------------------
CHIPS = ("wildcard", "freehit", "bboost", "3xc")        # names used by the FPL API
CHIP_LABEL = {"wildcard": "Wildcard", "freehit": "Free Hit",
              "bboost": "Bench Boost", "3xc": "Triple Captain"}
FIRST_HALF_LAST_GW = 19
SECOND_HALF_FIRST_GW = 20
LAST_GW = 38

# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
GOAL_PTS = {1: 10, 2: 6, 3: 5, 4: 4}
ASSIST_PTS = 3
CS_PTS = {1: 4, 2: 4, 3: 1, 4: 0}
YELLOW_PTS = -1
RED_PTS = -3
DC_POINTS = 2
# Defensive-contribution thresholds (2025/26 values).  ASSUMPTION: unchanged
# for 2026/27 -- the official announcement did not list new numbers.
# Run scripts/check_schema.py to see the live distributions and sanity-check.
DC_THRESHOLD = {2: 10, 3: 12, 4: 12}


@dataclass
class ModelParams:
    """Tuning knobs.  Defaults are deliberately conservative: long-run
    underlying numbers dominate and recent form gets a small, capped say."""

    # horizon / weighting
    horizon: int = 8                  # gameweeks looked ahead for squad decisions
    gw_decay: float = 0.97            # later gameweeks count slightly less
    bench_weight: float = 0.10        # value of a benched player's points (auto-subs)

    # rate estimation
    xg_weight: float = 0.75           # underlying (xG/xA) vs actual goals/assists
    rate_prior_minutes: float = 600   # pseudo-minutes of positional-average prior
    bonus_prior_minutes: float = 900
    recent_weight: float = 0.15       # CAP on how much last-N-games form can move a rate
    recent_window: int = 6
    recent_prior_minutes: float = 360
    past_season_prior_scale: float = 1500   # minutes at which last season gets ~50% prior weight

    # minutes / availability
    minutes_window: int = 8
    minutes_recent_weight: float = 0.5
    minutes_prior_rows: float = 4.0
    injury_recovery_gws: float = 6.0       # ASSUMPTION: unknown-return injuries heal linearly
    doubtful_recovery_gws: float = 2.0
    suspension_gws: int = 2                # ASSUMPTION
    dgw_play_factor: float = 0.95          # rotation risk per fixture in a double

    # team strength
    team_prior_games: float = 8.0
    team_strength_exponent: float = 1.5

    # set pieces (floors, not additions, to avoid double counting xG/xA)
    pen_taker_goal_floor: float = 0.09     # goals/90 above the positional prior
    fk_taker_goal_bonus: float = 0.02
    corner_taker_assist_bonus: float = 0.04

    # transfer discipline (anti "chasing last week's points")
    switch_cost: float = 1.0               # points a transfer must clear to be worth the churn
    roll_value: float = 1.5                # value of banking one extra free transfer
    hit_margin: float = 2.0                # extra points a -4 hit must clear beyond its cost
    min_robust_gain: float = 0.0           # gain must also hold under the long-term-only model
    max_transfers_considered: int = 3
