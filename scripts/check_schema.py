#!/usr/bin/env python
"""Run this FIRST on your own machine.

The model was developed against a synthetic league because the FPL API wasn't
reachable from the build environment.  This script hits the real API and checks
that every field the code relies on exists, then prints a few sanity numbers
(notably the defensive-contribution distribution, to confirm the thresholds in
model/config.py).

    python scripts/check_schema.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from etl.fpl_api import FPLAPIError, FPLClient   # noqa: E402

ELEMENT_FIELDS = [
    "id", "code", "web_name", "team", "element_type", "now_cost", "cost_change_start", "status",
    "chance_of_playing_next_round", "news", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "saves", "bonus", "yellow_cards", "red_cards",
    "expected_goals", "expected_assists", "expected_goals_conceded", "defensive_contribution",
    "total_points", "selected_by_percent", "penalties_order", "direct_freekicks_order",
    "corners_and_indirect_freekicks_order"]
TEAM_FIELDS = ["id", "name", "short_name", "strength_attack_home", "strength_attack_away",
               "strength_defence_home", "strength_defence_away"]
EVENT_FIELDS = ["id", "deadline_time", "finished", "is_current", "is_next"]
HISTORY_FIELDS = ["fixture", "round", "minutes", "goals_scored", "assists", "expected_goals",
                  "expected_assists", "defensive_contribution", "total_points", "was_home"]
FIXTURE_FIELDS = ["id", "event", "team_h", "team_a", "finished", "team_h_score", "team_a_score"]


def missing(obj: dict, fields):
    return [f for f in fields if f not in obj]


def main() -> int:
    c = FPLClient(delay=0.1)
    problems = 0
    bs = c.bootstrap()
    for label, rows, fields in (("elements", bs["elements"], ELEMENT_FIELDS),
                                ("teams", bs["teams"], TEAM_FIELDS),
                                ("events", bs["events"], EVENT_FIELDS)):
        m = missing(rows[0], fields)
        print(f"{label:9s}: {len(rows):4d} rows, missing fields: {m or 'none'}")
        problems += bool(m)

    fx = c.fixtures()
    m = missing(fx[0], FIXTURE_FIELDS)
    print(f"fixtures : {len(fx):4d} rows, missing fields: {m or 'none'}")
    problems += bool(m)

    nxt = [e for e in bs["events"] if e.get("is_next")]
    cur = [e for e in bs["events"] if e.get("is_current")]
    print(f"current GW: {cur[0]['id'] if cur else None} | next GW: {nxt[0]['id'] if nxt else None}")
    print("element types:", [(t["id"], t.get("plural_name_short")) for t in bs.get("element_types", [])])

    # Per-player history: sample a few starters
    starters = sorted((e for e in bs["elements"] if e["minutes"] > 600),
                      key=lambda e: -e["minutes"])[:40]
    if starters:
        summ = c.element_summary(starters[0]["id"])
        hist = summ.get("history", [])
        if hist:
            m = missing(hist[0], HISTORY_FIELDS)
            print(f"history  : {len(hist)} rows for {starters[0]['web_name']}, missing fields: {m or 'none'}")
            problems += bool(m)
        print("history_past keys:", sorted(summ["history_past"][0].keys())[:12] if summ.get("history_past") else "none")

        # Defensive-contribution sanity: how often do outfield players reach 10 / 12?
        by_pos = {2: [], 3: [], 4: []}
        for e in starters:
            if e["element_type"] in by_pos:
                for h in c.element_summary(e["id"]).get("history", []):
                    if h["minutes"] > 0:
                        by_pos[e["element_type"]].append(h.get("defensive_contribution", 0))
        for pos, vals in by_pos.items():
            if vals:
                dist = Counter(min(v, 15) for v in vals)
                thr = {2: 10, 3: 12, 4: 12}[pos]
                hit = sum(v >= thr for v in vals) / len(vals)
                print(f"DC pos {pos}: n={len(vals)}, mean={sum(vals)/len(vals):.1f}, "
                      f"share >= threshold {thr}: {hit:.0%}  (sanity range roughly 15-50% for DEF, 5-25% MID/FWD)")

    print("\nRESULT:", "all required fields present" if not problems
          else f"{problems} section(s) have missing fields -- the schema changed; "
               "fix etl/pipeline.py (and tell me what's missing).")
    return 1 if problems else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except FPLAPIError as exc:
        print(f"Could not reach the FPL API: {exc}\nCheck your internet connection / try again later.")
        sys.exit(2)
