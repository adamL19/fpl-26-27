"""Expected-points projections per player per gameweek.

Design goals (from the brief): points that relate directly to FPL scoring, and
resistance to short-term noise.  So:

* Goal / assist rates come mostly from season-long *expected* goals/assists
  (xG/xA), shrunk toward a positional (and last-season) prior by minutes played.
* Recent form may move a rate only a little (`recent_weight`, default 15%).
* Playing time is modelled separately (starts, minutes when playing, injury news).
* Fixtures matter through a Poisson team-strength model (attack multiplier for
  goals/assists, expected goals against for clean sheets, goals conceded, saves).
* Penalty / free-kick / corner takers get rate floors; defensive-contribution
  points come from each player's own hit-rate over the season.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.config import (ASSIST_PTS, CS_PTS, DC_POINTS, DC_THRESHOLD, GOAL_PTS,
                          LAST_GW, RED_PTS, YELLOW_PTS, ModelParams)
from model.team_ratings import TeamModel, fit_team_model

_N = np.arange(0, 45)
_FACT = np.cumprod(np.r_[1.0, np.arange(1, 45, dtype=float)])   # float: int64 overflows at 21!


def poisson_floor_expect(lam: float, k: int) -> float:
    """E[floor(X / k)] for X ~ Poisson(lam)."""
    if lam <= 0:
        return 0.0
    pmf = np.exp(-lam) * lam ** _N / _FACT
    return float((pmf * (_N // k)).sum())


# Prior for players with almost no history: (P(appear), minutes when appearing, P(>=60 | appear))
_MIN_PRIOR = {1: (0.90, 90.0, 1.0), 2: (0.65, 78.0, 0.75),
              3: (0.65, 70.0, 0.60), 4: (0.60, 68.0, 0.55)}


@dataclass
class Projections:
    long: pd.DataFrame          # player_id, gw, xpts, p_play, xmins, n_fix
    players: pd.DataFrame       # player-level rates (for debugging / dashboards)

    def matrix(self) -> pd.DataFrame:
        return self.long.pivot_table(index="player_id", columns="gw",
                                     values="xpts", aggfunc="sum", fill_value=0.0)


# --------------------------------------------------------------------------
# Positional priors
# --------------------------------------------------------------------------
def positional_priors(players: pd.DataFrame, history: pd.DataFrame,
                      params: ModelParams) -> dict:
    out: dict = {}
    for pos in (1, 2, 3, 4):
        sub = players[(players["position"] == pos) & (players["minutes"] >= 450)]
        m90 = sub["minutes"].sum() / 90.0
        if m90 <= 0 or sub.empty:
            out[pos] = dict(g90=0.05 * pos / 4, a90=0.05, bonus90=0.12, saves90=0.0,
                            yc90=0.12, rc90=0.004, dc_hit=0.25)
            if pos == 1:
                out[pos]["saves90"] = 2.8
            continue
        obs_g = params.xg_weight * sub["xg"].sum() + (1 - params.xg_weight) * sub["goals"].sum()
        obs_a = params.xg_weight * sub["xa"].sum() + (1 - params.xg_weight) * sub["assists"].sum()
        out[pos] = dict(
            g90=obs_g / m90, a90=obs_a / m90, bonus90=sub["bonus"].sum() / m90,
            saves90=sub["saves"].sum() / m90, yc90=sub["yellow"].sum() / m90,
            rc90=sub["red"].sum() / m90, dc_hit=0.25)
        if pos in DC_THRESHOLD and not history.empty:
            ids = set(players.loc[players["position"] == pos, "id"])
            h = history[history["player_id"].isin(ids) & (history["minutes"] > 0)]
            if len(h) >= 30:
                out[pos]["dc_hit"] = float((h["dc"] >= DC_THRESHOLD[pos]).mean())
    return out


# --------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------
def availability(status: str, chance_next, offset: int, params: ModelParams) -> float:
    """Probability the player is fit to play in the GW `offset` weeks after the next one."""
    c = None if chance_next is None or (isinstance(chance_next, float) and np.isnan(chance_next)) \
        else float(chance_next) / 100.0
    if status in ("u", "n"):
        return 0.0
    if status == "s":
        return 0.0 if offset < params.suspension_gws else 1.0
    if status == "d":
        base = 0.5 if c is None else c
        return min(1.0, base + (1 - base) * min(1.0, offset / params.doubtful_recovery_gws))
    if status == "i":
        base = 0.0 if c is None else c
        return min(1.0, base + (1 - base) * min(1.0, offset / params.injury_recovery_gws))
    # 'a'
    if c is None:
        return 1.0
    return min(1.0, c + (1 - c) * min(1.0, offset / params.doubtful_recovery_gws))


# --------------------------------------------------------------------------
# Per-player rate estimation
# --------------------------------------------------------------------------
def _minutes_profile(rows: pd.DataFrame, status: str, pos: int, params: ModelParams):
    """(P(appear | fit), E[minutes | appear], P(>=60 | appear))"""
    p0, m0, s0 = _MIN_PRIOR[pos]
    if rows.empty:
        return p0, m0, s0
    mins = rows["minutes"].to_numpy(float)
    # If currently unavailable, trailing zero-minute rows are most likely the injury itself
    if status != "a":
        k = 0
        while k < len(mins) and mins[-1 - k] == 0:
            k += 1
        if k:
            mins = mins[:-k]
    if len(mins) == 0:
        return p0, m0, s0

    def stats(arr, weights=None):
        w = np.ones(len(arr)) if weights is None else weights
        played = arr > 0
        app = (w * played).sum() / w.sum()
        if (w * played).sum() > 0:
            mga = (w * arr * played).sum() / (w * played).sum()
            s60 = (w * (arr >= 60) * played).sum() / (w * played).sum()
        else:
            mga, s60 = np.nan, np.nan
        return app, mga, s60

    season = stats(mins)
    recent_arr = mins[-params.minutes_window:]
    wts = 0.85 ** np.arange(len(recent_arr))[::-1]
    recent = stats(recent_arr, wts)
    mr = params.minutes_recent_weight
    out = []
    for i in range(3):
        s, r = season[i], recent[i]
        if np.isnan(s) and np.isnan(r):
            out.append(np.nan)
        elif np.isnan(r):
            out.append(s)
        elif np.isnan(s):
            out.append(r)
        else:
            out.append(mr * r + (1 - mr) * s)
    n = len(mins)
    w = n / (n + params.minutes_prior_rows)
    prior = (p0, m0, s0)
    res = []
    for i in range(3):
        v = prior[i] if np.isnan(out[i]) else w * out[i] + (1 - w) * prior[i]
        res.append(v)
    return float(res[0]), float(res[1]), float(res[2])


def _rate(total_obs: float, minutes: float, prior_rate: float, prior_minutes: float) -> float:
    m90, k90 = minutes / 90.0, prior_minutes / 90.0
    return (total_obs + k90 * prior_rate) / (m90 + k90)


def player_rates(p: pd.Series, rows: pd.DataFrame, past: pd.DataFrame,
                 priors: dict, params: ModelParams) -> dict:
    pos = int(p["position"])
    pr = priors[pos]
    mins = float(p["minutes"] or 0)

    # --- prior: positional average blended with the last two seasons (if any)
    g_prior, a_prior = pr["g90"], pr["a90"]
    if not past.empty:
        last = past.sort_values("season_name").tail(2)
        pm = float(last["minutes"].sum())
        if pm > 0:
            use_x = last["xg"].sum() > 0
            pg = (params.xg_weight * last["xg"].sum() + (1 - params.xg_weight) * last["goals"].sum()
                  if use_x else last["goals"].sum())
            pa = (params.xg_weight * last["xa"].sum() + (1 - params.xg_weight) * last["assists"].sum()
                  if use_x else last["assists"].sum())
            w = pm / (pm + params.past_season_prior_scale)
            g_prior = w * (pg / (pm / 90)) + (1 - w) * g_prior
            a_prior = w * (pa / (pm / 90)) + (1 - w) * a_prior

    obs_g = params.xg_weight * p["xg"] + (1 - params.xg_weight) * p["goals"]
    obs_a = params.xg_weight * p["xa"] + (1 - params.xg_weight) * p["assists"]
    g90 = _rate(obs_g, mins, g_prior, params.rate_prior_minutes)
    a90 = _rate(obs_a, mins, a_prior, params.rate_prior_minutes)

    # --- recent form: capped, and itself shrunk toward the long-run rate
    if params.recent_weight > 0 and not rows.empty:
        rec = rows.tail(params.recent_window)
        rm = float(rec["minutes"].sum())
        if rm > 0:
            rg = _rate(params.xg_weight * rec["xg"].sum() + (1 - params.xg_weight) * rec["goals"].sum(),
                       rm, g90, params.recent_prior_minutes)
            ra = _rate(params.xg_weight * rec["xa"].sum() + (1 - params.xg_weight) * rec["assists"].sum(),
                       rm, a90, params.recent_prior_minutes)
            g90 = (1 - params.recent_weight) * g90 + params.recent_weight * rg
            a90 = (1 - params.recent_weight) * a90 + params.recent_weight * ra

    # --- set-piece floors
    if p.get("pen_order") == 1:
        g90 = max(g90, pr["g90"] + params.pen_taker_goal_floor)
    if p.get("fk_order") == 1:
        g90 += params.fk_taker_goal_bonus * (1 if g90 < pr["g90"] + 0.15 else 0.5)
    if p.get("corner_order") == 1:
        a90 = max(a90, pr["a90"] + params.corner_taker_assist_bonus)

    bonus90 = _rate(p["bonus"], mins, pr["bonus90"], params.bonus_prior_minutes)
    saves90 = _rate(p["saves"], mins, pr["saves90"], params.bonus_prior_minutes) if pos == 1 else 0.0
    yc90 = _rate(p["yellow"], mins, pr["yc90"], params.bonus_prior_minutes)
    rc90 = _rate(p["red"], mins, pr["rc90"], params.bonus_prior_minutes * 2)

    # --- defensive contribution hit rate (season-long, shrunk by appearances)
    dc_hit = 0.0
    if pos in DC_THRESHOLD:
        played = rows[rows["minutes"] > 0] if not rows.empty else rows
        n = len(played)
        hits = float((played["dc"] >= DC_THRESHOLD[pos]).sum()) if n else 0.0
        k = 6.0
        dc_hit = (hits + k * pr["dc_hit"]) / (n + k)

    app, mga, s60 = _minutes_profile(rows, p["status"], pos, params)
    return dict(g90=g90, a90=a90, bonus90=bonus90, saves90=saves90, yc90=yc90, rc90=rc90,
                dc_hit=dc_hit, p_appear=app, mins_app=mga, p60=s60)


# --------------------------------------------------------------------------
# Fixture-level expected points
# --------------------------------------------------------------------------
def fixture_points(pos: int, r: dict, team: int, opp: int, is_home: bool,
                   tm: TeamModel) -> float:
    """Expected points *given the player appears* in one fixture."""
    lam_for, lam_against = tm.lam(team, opp, is_home)
    mu_v = tm.mu_home if is_home else tm.mu_away
    mu_o = tm.mu_away if is_home else tm.mu_home
    m_att = (mu_v / tm.mu_bar) * tm.dfn[opp]            # attack multiplier vs this opponent
    m_opp_att = (mu_o / tm.mu_bar) * tm.att[opp]        # how hard the opponent attacks (saves)
    mins = r["mins_app"]
    frac = mins / 90.0

    pts = 1.0 + r["p60"]                                           # appearance points
    pts += GOAL_PTS[pos] * r["g90"] * m_att * frac
    pts += ASSIST_PTS * r["a90"] * m_att * frac
    pts += (r["bonus90"] * frac *
            (np.sqrt(m_att) if pos in (3, 4)
             else float(np.clip((tm.mu_bar * tm.dfn[team] / max(lam_against, 1e-6)) ** 0.25, 0.7, 1.4))))
    pts += YELLOW_PTS * r["yc90"] * frac + RED_PTS * r["rc90"] * frac

    if pos in (1, 2):
        mins60 = float(np.clip(mins, 60, 90))
        pts += CS_PTS[pos] * r["p60"] * np.exp(-lam_against * mins60 / 90.0)
        pts -= poisson_floor_expect(lam_against * frac, 2)
    elif pos == 3:
        pts += CS_PTS[3] * r["p60"] * np.exp(-lam_against * float(np.clip(mins, 60, 90)) / 90.0)
    if pos == 1:
        pts += poisson_floor_expect(r["saves90"] * m_opp_att * frac, 3)
    if pos in DC_THRESHOLD:
        pts += DC_POINTS * r["dc_hit"]
    return float(pts)


def build_projections(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame,
                      history: pd.DataFrame, past: pd.DataFrame, gws: list[int],
                      params: ModelParams | None = None) -> Projections:
    """Expected points for every player in every gameweek of `gws`."""
    params = params or ModelParams()
    tm = fit_team_model(teams, fixtures, params)
    priors = positional_priors(players, history, params)

    sched: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    fx = fixtures.dropna(subset=["event"])
    for f in fx.itertuples():
        g = int(f.event)
        sched.setdefault((int(f.team_h), g), []).append((int(f.team_a), True))
        sched.setdefault((int(f.team_a), g), []).append((int(f.team_h), False))

    next_gw = min(gws)
    hist_by_player = {pid: df.sort_values(["kickoff_time", "round"])
                      for pid, df in history.groupby("player_id")} if not history.empty else {}
    past_by_player = {pid: df for pid, df in past.groupby("player_id")} if not past.empty else {}
    empty_h = pd.DataFrame(columns=history.columns) if not history.empty else pd.DataFrame(
        columns=["minutes", "xg", "xa", "goals", "assists", "dc"])
    empty_p = pd.DataFrame(columns=["season_name", "minutes", "goals", "assists", "xg", "xa"])

    long_rows, rate_rows = [], []
    for _, p in players.iterrows():
        pid = int(p["id"]); pos = int(p["position"]); team = int(p["team_id"])
        rows = hist_by_player.get(pid, empty_h)
        r = player_rates(p, rows, past_by_player.get(pid, empty_p), priors, params)
        rate_rows.append({"player_id": pid, **r})
        for g in gws:
            fl = sched.get((team, g), [])
            if not fl:
                long_rows.append((pid, g, 0.0, 0.0, 0.0, 0))
                continue
            avail = availability(p["status"], p["chance_next"], g - next_gw, params)
            play_f = params.dgw_play_factor if len(fl) > 1 else 1.0
            p_play = avail * r["p_appear"] * play_f
            xp = sum(p_play * fixture_points(pos, r, team, opp, home, tm) for opp, home in fl)
            long_rows.append((pid, g, xp, p_play, p_play * r["mins_app"] * len(fl), len(fl)))
    long = pd.DataFrame(long_rows, columns=["player_id", "gw", "xpts", "p_play", "xmins", "n_fix"])
    return Projections(long=long, players=pd.DataFrame(rate_rows))


def gw_window(next_gw: int, n: int) -> list[int]:
    return list(range(next_gw, min(next_gw + n, LAST_GW + 1)))
