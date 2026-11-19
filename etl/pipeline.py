"""ETL: FPL API -> SQLite.

Parsing functions are pure (JSON in, DataFrame out) so they can be tested with
canned data.  `refresh()` wires them to a client and a database connection.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

import pandas as pd

from db.database import replace_table


def _f(x, default=0.0) -> float:
    """API sometimes returns numbers as strings ('0.45') or None."""
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _opt(x):
    try:
        return None if x is None or x == "" else float(x)
    except (TypeError, ValueError):
        return None


def _dc_total(row: dict, position: int) -> float:
    """Defensive-contribution count.  Uses the API field when present, else
    rebuilds it from components (CBIT for defenders, CBIRT for others)."""
    if "defensive_contribution" in row and row["defensive_contribution"] is not None:
        return _f(row["defensive_contribution"])
    cbi = _f(row.get("clearances_blocks_interceptions"))
    tck = _f(row.get("tackles"))
    rec = _f(row.get("recoveries"))
    if position == 1:
        return 0.0
    return cbi + tck + (rec if position in (3, 4) else 0.0)


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------
def parse_players(bootstrap: dict) -> pd.DataFrame:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for e in bootstrap["elements"]:
        pos = int(e["element_type"])
        rows.append({
            "id": e["id"], "code": e.get("code"),
            "web_name": e.get("web_name"), "first_name": e.get("first_name"),
            "second_name": e.get("second_name"),
            "team_id": e["team"], "position": pos,
            "now_cost": e["now_cost"],
            "start_cost": e["now_cost"] - e.get("cost_change_start", 0),
            "status": e.get("status", "a"),
            "chance_next": _opt(e.get("chance_of_playing_next_round")),
            "chance_this": _opt(e.get("chance_of_playing_this_round")),
            "news": e.get("news") or "",
            "minutes": e.get("minutes", 0), "starts": e.get("starts", 0),
            "goals": e.get("goals_scored", 0), "assists": e.get("assists", 0),
            "clean_sheets": e.get("clean_sheets", 0),
            "goals_conceded": e.get("goals_conceded", 0),
            "saves": e.get("saves", 0), "bonus": e.get("bonus", 0),
            "yellow": e.get("yellow_cards", 0), "red": e.get("red_cards", 0),
            "own_goals": e.get("own_goals", 0),
            "pens_saved": e.get("penalties_saved", 0),
            "pens_missed": e.get("penalties_missed", 0),
            "xg": _f(e.get("expected_goals")), "xa": _f(e.get("expected_assists")),
            "xgi": _f(e.get("expected_goal_involvements")),
            "xgc": _f(e.get("expected_goals_conceded")),
            "dc": _dc_total(e, pos),
            "total_points": e.get("total_points", 0),
            "selected_pct": _f(e.get("selected_by_percent")),
            "form": _f(e.get("form")),
            "ep_next": _f(e.get("ep_next")),
            "pen_order": _opt(e.get("penalties_order")),
            "fk_order": _opt(e.get("direct_freekicks_order")),
            "corner_order": _opt(e.get("corners_and_indirect_freekicks_order")),
            "fetched_at": now,
        })
    return pd.DataFrame(rows)


def parse_teams(bootstrap: dict) -> pd.DataFrame:
    cols = ["id", "name", "short_name", "strength", "strength_attack_home",
            "strength_attack_away", "strength_defence_home", "strength_defence_away"]
    return pd.DataFrame([{c: t.get(c) for c in cols} for t in bootstrap["teams"]])


def parse_events(bootstrap: dict) -> pd.DataFrame:
    rows = []
    for ev in bootstrap["events"]:
        rows.append({"id": ev["id"], "name": ev.get("name"),
                     "deadline_time": ev.get("deadline_time"),
                     "finished": int(bool(ev.get("finished"))),
                     "is_current": int(bool(ev.get("is_current"))),
                     "is_next": int(bool(ev.get("is_next"))),
                     "data_checked": int(bool(ev.get("data_checked")))})
    return pd.DataFrame(rows)


def parse_fixtures(fixtures: list) -> pd.DataFrame:
    rows = []
    for f in fixtures:
        rows.append({"id": f["id"], "event": f.get("event"),
                     "team_h": f["team_h"], "team_a": f["team_a"],
                     "kickoff_time": f.get("kickoff_time"),
                     "finished": int(bool(f.get("finished"))),
                     "started": int(bool(f.get("started"))),
                     "team_h_score": f.get("team_h_score"),
                     "team_a_score": f.get("team_a_score"),
                     "team_h_difficulty": f.get("team_h_difficulty"),
                     "team_a_difficulty": f.get("team_a_difficulty")})
    return pd.DataFrame(rows)


def parse_element_summary(player_id: int, position: int, summary: dict,
                          code_to_id: dict[int, int] | None = None):
    """Returns (gw_history_df, past_seasons_df) for one player."""
    hist = []
    for h in summary.get("history", []):
        hist.append({
            "player_id": player_id, "fixture": h.get("fixture"), "round": h.get("round"),
            "opponent_team": h.get("opponent_team"), "was_home": int(bool(h.get("was_home"))),
            "kickoff_time": h.get("kickoff_time"), "minutes": h.get("minutes", 0),
            "starts": h.get("starts", 1 if h.get("minutes", 0) >= 45 else 0),
            "goals": h.get("goals_scored", 0), "assists": h.get("assists", 0),
            "clean_sheets": h.get("clean_sheets", 0),
            "goals_conceded": h.get("goals_conceded", 0),
            "saves": h.get("saves", 0), "bonus": h.get("bonus", 0), "bps": h.get("bps", 0),
            "yellow": h.get("yellow_cards", 0), "red": h.get("red_cards", 0),
            "own_goals": h.get("own_goals", 0),
            "xg": _f(h.get("expected_goals")), "xa": _f(h.get("expected_assists")),
            "xgi": _f(h.get("expected_goal_involvements")),
            "xgc": _f(h.get("expected_goals_conceded")),
            "dc": _dc_total(h, position),
            "total_points": h.get("total_points", 0), "value": h.get("value"),
        })
    past = []
    for p in summary.get("history_past", []):
        past.append({"player_id": player_id, "season_name": p.get("season_name"),
                     "start_cost": p.get("start_cost"), "end_cost": p.get("end_cost"),
                     "total_points": p.get("total_points", 0), "minutes": p.get("minutes", 0),
                     "goals": p.get("goals_scored", 0), "assists": p.get("assists", 0),
                     "xg": _f(p.get("expected_goals")), "xa": _f(p.get("expected_assists"))})
    return pd.DataFrame(hist), pd.DataFrame(past)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def refresh(client, conn, fetch_histories: bool = True, all_players: bool = False,
            min_selected_pct: float = 1.0, progress: Callable[[str], None] = print) -> dict:
    """Pull everything we need and write it to SQLite.

    By default per-player histories are fetched only for players who have
    played or are owned by >= `min_selected_pct`% of managers (keeps the
    number of API calls to a few hundred)."""
    progress("Fetching bootstrap-static ...")
    bs = client.bootstrap()
    players = parse_players(bs)
    replace_table(conn, "players", players)
    replace_table(conn, "teams", parse_teams(bs))
    replace_table(conn, "events", parse_events(bs))

    progress("Fetching fixtures ...")
    replace_table(conn, "fixtures", parse_fixtures(client.fixtures()))

    n_hist = 0
    if fetch_histories:
        todo = players if all_players else players[
            (players["minutes"] > 0) | (players["selected_pct"] >= min_selected_pct)]
        total = len(todo)
        progress(f"Fetching per-player history for {total} players ...")
        failures = []
        for i, (_, p) in enumerate(todo.iterrows(), 1):
            try:
                summary = client.element_summary(int(p["id"]))
            except Exception as exc:                       # keep going, report at the end
                failures.append((int(p["id"]), str(exc)))
                continue
            h, past = parse_element_summary(int(p["id"]), int(p["position"]), summary)
            replace_table(conn, "gw_history", h, where="player_id = ?", params=(int(p["id"]),))
            replace_table(conn, "past_seasons", past, where="player_id = ?", params=(int(p["id"]),))
            n_hist += 1
            if i % 100 == 0:
                progress(f"  ... {i}/{total}")
        if failures:
            progress(f"WARNING: {len(failures)} player histories failed "
                     f"(e.g. {failures[0]}). Re-run refresh to retry.")
    return {"players": len(players), "histories": n_hist}
