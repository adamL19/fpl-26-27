"""Synthetic FPL league that mimics the shape of the real API responses.

Used only for tests / offline development.  It exposes the same methods as
etl.fpl_api.FPLClient so the real pipeline code runs unchanged.
"""
from __future__ import annotations

import numpy as np

POS_QUOTA_TEAM = {1: 2, 2: 7, 3: 8, 4: 4}
PRICE_RANGE = {1: (40, 62), 2: (40, 75), 3: (45, 130), 4: (45, 140)}
G90 = {1: 0.0, 2: 0.05, 3: 0.15, 4: 0.40}
A90 = {1: 0.0, 2: 0.07, 3: 0.15, 4: 0.15}
DC_MEAN = {1: 0.0, 2: 9.0, 3: 9.0, 4: 5.0}


def _round_robin(n_teams: int) -> list[list[tuple[int, int]]]:
    teams = list(range(1, n_teams + 1))
    rounds = []
    for r in range(n_teams - 1):
        pairs = []
        for i in range(n_teams // 2):
            a, b = teams[i], teams[-1 - i]
            pairs.append((a, b) if (r + i) % 2 == 0 else (b, a))
        rounds.append(pairs)
        teams = [teams[0]] + [teams[-1]] + teams[1:-1]
    return rounds + [[(b, a) for a, b in rnd] for rnd in rounds]


class FakeFPL:
    ENTRY_ID = 424242

    def __init__(self, seed: int = 7, current_gw: int = 11):
        self.rng = np.random.default_rng(seed)
        self.current_gw = current_gw
        self._build_teams()
        self._build_fixtures()
        self._build_players()
        self._simulate()
        self._build_manager()

    # ------------------------------------------------------------------
    def _build_teams(self):
        r = self.rng
        self.t_att = np.exp(r.normal(0, 0.18, 20)); self.t_def = np.exp(r.normal(0, 0.18, 20))
        self.teams = []
        for i in range(20):
            s_att = int(1150 + 120 * np.log(self.t_att[i]) / 0.18 * 0.5)
            s_def = int(1150 - 120 * np.log(self.t_def[i]) / 0.18 * 0.5)
            self.teams.append({"id": i + 1, "name": f"Team {i+1}", "short_name": f"T{i+1:02d}",
                               "strength": 3, "strength_attack_home": s_att + 30,
                               "strength_attack_away": s_att - 30,
                               "strength_defence_home": s_def + 30, "strength_defence_away": s_def - 30})

    def _build_fixtures(self):
        rounds = _round_robin(20)
        sched = {g + 1: list(p) for g, p in enumerate(rounds)}
        # Blank GW25 for 8 teams (4 matches), moved into GW30 => 8 teams double there.
        moved = sched[25][:4]
        sched[25] = sched[25][4:]
        sched[30] = sched[30] + moved
        # Two matches left unscheduled in GW33
        unsched = sched[33][:2]
        sched[33] = sched[33][2:]
        fid = 1
        self.fixtures_raw = []
        for g in range(1, 39):
            for h, a in sched[g]:
                self.fixtures_raw.append({"id": fid, "event": g, "team_h": h, "team_a": a,
                                          "kickoff_time": f"2026-{8 + (g-1)//5:02d}-{1 + (g-1)%5*5:02d}T15:00:00Z",
                                          "finished": g <= self.current_gw, "started": g <= self.current_gw,
                                          "team_h_score": None, "team_a_score": None,
                                          "team_h_difficulty": 3, "team_a_difficulty": 3})
                fid += 1
        for h, a in unsched:
            self.fixtures_raw.append({"id": fid, "event": None, "team_h": h, "team_a": a,
                                      "kickoff_time": None, "finished": False, "started": False,
                                      "team_h_score": None, "team_a_score": None,
                                      "team_h_difficulty": 3, "team_a_difficulty": 3})
            fid += 1

    def _build_players(self):
        r = self.rng
        pid = 1
        raw = []
        for t in range(1, 21):
            for pos, n in POS_QUOTA_TEAM.items():
                for k in range(n):
                    q = float(np.exp(r.normal(0, 0.45)))
                    p_start = float(np.clip(r.beta(5, 2) if k < {1: 1, 2: 4, 3: 4, 4: 2}[pos] else r.beta(1.5, 6), 0.02, 0.97))
                    raw.append(dict(id=pid, team=t, pos=pos, q=q, p_start=p_start,
                                    code=100000 + pid))
                    pid += 1
        for pos in (1, 2, 3, 4):
            sub = [p for p in raw if p["pos"] == pos]
            ability = np.array([p["q"] * (0.5 + p["p_start"]) for p in sub])
            pct = ability.argsort().argsort() / (len(sub) - 1)
            lo, hi = PRICE_RANGE[pos]
            for p, pc in zip(sub, pct):
                p["start_cost"] = int(round((lo + (hi - lo) * pc ** 1.8) / 5) * 5)
        self.pl = {p["id"]: p for p in raw}
        # Set-piece takers: best attacker per team has penalties, etc.
        for t in range(1, 21):
            att = sorted([p for p in raw if p["team"] == t and p["pos"] in (3, 4)], key=lambda p: -p["q"])
            att[0]["pen"] = 1; att[1]["pen"] = 2
            att[min(2, len(att) - 1)]["fk"] = 1
            att[1]["corner"] = 1
        # Injuries / news
        for p in r.choice(raw, 25, replace=False):
            kind = r.choice(["i", "d", "s", "u"], p=[0.45, 0.35, 0.15, 0.05])
            p["status"] = kind
            p["chance"] = {"i": 0, "d": 50, "s": 0, "u": 0}[kind] if kind != "i" or r.random() < 0.5 else None
            if kind == "d":
                p["chance"] = 75 if r.random() < 0.5 else 50
        # A couple of players who changed price
        for p in raw:
            p["now_cost"] = p["start_cost"] + int(r.choice([-5, 0, 0, 0, 5, 5, 10])) \
                if p["start_cost"] > 45 else p["start_cost"]

    def _simulate(self):
        r = self.rng
        done = [f for f in self.fixtures_raw if f["event"] and f["event"] <= self.current_gw]
        mu = {True: 1.5, False: 1.2}
        for f in done:
            h, a = f["team_h"] - 1, f["team_a"] - 1
            lam_h = mu[True] * self.t_att[h] * self.t_def[a]; lam_a = mu[False] * self.t_att[a] * self.t_def[h]
            f["team_h_score"] = int(r.poisson(lam_h)); f["team_a_score"] = int(r.poisson(lam_a))
            f["_lam"] = (lam_h, lam_a)
        self.history = {p: [] for p in self.pl}
        for f in done:
            for home in (True, False):
                team = f["team_h"] if home else f["team_a"]
                opp = f["team_a"] if home else f["team_h"]
                gf = f["team_h_score"] if home else f["team_a_score"]
                ga = f["team_a_score"] if home else f["team_h_score"]
                lam_f, lam_a = f["_lam"] if home else f["_lam"][::-1]
                avg_f = 1.35 * self.t_att[team - 1]
                for p in [p for p in self.pl.values() if p["team"] == team]:
                    pos = p["pos"]
                    if p.get("status") in ("u",) and f["event"] > 4:
                        mins = 0
                    elif r.random() < p["p_start"]:
                        mins = 90 if r.random() < 0.7 else int(r.integers(60, 90))
                    else:
                        mins = int(r.integers(8, 30)) if r.random() < 0.3 else 0
                    fac = lam_f / avg_f if avg_f else 1.0
                    xg = G90[pos] * p["q"] * fac * mins / 90 * float(np.exp(r.normal(0, 0.2)))
                    xa = A90[pos] * p["q"] * fac * mins / 90 * float(np.exp(r.normal(0, 0.2)))
                    goals = int(r.poisson(xg * 1.0)); assists = int(r.poisson(xa))
                    dc = int(r.poisson(DC_MEAN[pos] * mins / 90)) if mins else 0
                    saves = int(r.poisson(3.0 * mins / 90 * lam_a / 1.35)) if pos == 1 and mins else 0
                    yc = int(r.random() < 0.08 * mins / 90); rc = int(r.random() < 0.003 * mins / 90)
                    bonus = int(r.choice([0, 0, 0, 1, 2, 3], p=[.5, .2, .1, .1, .06, .04])) if mins >= 45 else 0
                    cs = int(ga == 0 and mins >= 60)
                    pts = ((1 if mins > 0 else 0) + (1 if mins >= 60 else 0)
                           + goals * {1: 10, 2: 6, 3: 5, 4: 4}[pos] + assists * 3
                           + cs * {1: 4, 2: 4, 3: 1, 4: 0}[pos] - (ga // 2 if pos in (1, 2) and mins else 0)
                           + saves // 3 - yc - 3 * rc + bonus + 2 * (dc >= {1: 99, 2: 10, 3: 12, 4: 12}[pos]))
                    self.history[p["id"]].append({
                        "element": p["id"], "fixture": f["id"], "opponent_team": opp, "total_points": pts,
                        "was_home": home, "kickoff_time": f["kickoff_time"], "round": f["event"],
                        "minutes": mins, "goals_scored": goals, "assists": assists, "clean_sheets": cs,
                        "goals_conceded": ga if mins else 0, "own_goals": 0, "penalties_saved": 0,
                        "penalties_missed": 0, "yellow_cards": yc, "red_cards": rc, "saves": saves,
                        "bonus": bonus, "bps": 0, "starts": int(mins >= 60),
                        "expected_goals": f"{xg:.2f}", "expected_assists": f"{xa:.2f}",
                        "expected_goal_involvements": f"{xg + xa:.2f}", "expected_goals_conceded": f"{lam_a:.2f}",
                        "defensive_contribution": dc, "value": p["now_cost"]})

    # ------------------------------------------------------------------
    def _agg(self, pid, key, num=True):
        return sum(float(h[key]) for h in self.history[pid])

    def bootstrap(self):
        elements = []
        for pid, p in self.pl.items():
            h = self.history[pid]
            mins = sum(x["minutes"] for x in h)
            elements.append({
                "id": pid, "code": p["code"], "web_name": f"P{pid}", "first_name": "Fake",
                "second_name": f"Player{pid}", "team": p["team"], "element_type": p["pos"],
                "now_cost": p["now_cost"], "cost_change_start": p["now_cost"] - p["start_cost"],
                "status": p.get("status", "a"), "chance_of_playing_next_round": p.get("chance"),
                "chance_of_playing_this_round": p.get("chance"),
                "news": "Fake news" if p.get("status", "a") != "a" else "",
                "minutes": mins, "starts": sum(x["starts"] for x in h),
                "goals_scored": sum(x["goals_scored"] for x in h), "assists": sum(x["assists"] for x in h),
                "clean_sheets": sum(x["clean_sheets"] for x in h), "goals_conceded": sum(x["goals_conceded"] for x in h),
                "own_goals": 0, "penalties_saved": 0, "penalties_missed": 0,
                "yellow_cards": sum(x["yellow_cards"] for x in h), "red_cards": sum(x["red_cards"] for x in h),
                "saves": sum(x["saves"] for x in h), "bonus": sum(x["bonus"] for x in h),
                "expected_goals": f"{self._agg(pid, 'expected_goals'):.2f}",
                "expected_assists": f"{self._agg(pid, 'expected_assists'):.2f}",
                "expected_goal_involvements": f"{self._agg(pid, 'expected_goal_involvements'):.2f}",
                "expected_goals_conceded": f"{self._agg(pid, 'expected_goals_conceded'):.2f}",
                "defensive_contribution": sum(x["defensive_contribution"] for x in h),
                "total_points": sum(x["total_points"] for x in h),
                "selected_by_percent": f"{min(60, 100 * p['p_start'] * p['q'] / 4):.1f}",
                "form": f"{np.mean([x['total_points'] for x in h[-5:]]) if h else 0:.1f}", "ep_next": "3.0",
                "penalties_order": p.get("pen"), "direct_freekicks_order": p.get("fk"),
                "corners_and_indirect_freekicks_order": p.get("corner")})
        events = []
        for g in range(1, 39):
            events.append({"id": g, "name": f"Gameweek {g}", "deadline_time": f"2026-12-{(g % 28) + 1:02d}T10:00:00Z",
                           "finished": g <= self.current_gw, "is_current": g == self.current_gw,
                           "is_next": g == self.current_gw + 1, "data_checked": g <= self.current_gw})
        return {"elements": elements, "teams": self.teams, "events": events,
                "element_types": [{"id": i} for i in (1, 2, 3, 4)]}

    def fixtures(self):
        return [{k: v for k, v in f.items() if not k.startswith("_")} for f in self.fixtures_raw]

    def element_summary(self, pid):
        past = [{"season_name": "2024/25", "element_code": self.pl[pid]["code"], "start_cost": self.pl[pid]["start_cost"],
                 "end_cost": self.pl[pid]["start_cost"], "total_points": 100, "minutes": 2000 if self.pl[pid]["p_start"] > 0.5 else 400,
                 "goals_scored": int(G90[self.pl[pid]["pos"]] * self.pl[pid]["q"] * 22),
                 "assists": int(A90[self.pl[pid]["pos"]] * self.pl[pid]["q"] * 22),
                 "expected_goals": "0.00", "expected_assists": "0.00"}]
        return {"fixtures": [], "history": self.history[pid], "history_past": past}

    # ------------------------------------------------------------------
    def _build_manager(self):
        """A plausible squad: sample until valid and <= 100.0m at current prices."""
        r = self.rng
        by_pos = {pos: [p for p in self.pl.values() if p["pos"] == pos and p.get("status", "a") == "a"] for pos in (1, 2, 3, 4)}
        for _ in range(5000):
            squad, counts = [], {}
            for pos, q in {1: 2, 2: 5, 3: 5, 4: 3}.items():
                squad += list(r.choice(by_pos[pos], q, replace=False))
            counts = {}
            for p in squad:
                counts[p["team"]] = counts.get(p["team"], 0) + 1
            cost = sum(p["now_cost"] for p in squad)
            if max(counts.values()) <= 3 and 960 <= cost <= 1000:
                break
        self.my_squad = [p["id"] for p in squad]
        # two earlier transfers into the squad
        self.my_transfers = [
            {"element_in": self.my_squad[7], "element_in_cost": self.pl[self.my_squad[7]]["now_cost"] - 10,
             "element_out": 9999, "element_out_cost": 50, "entry": self.ENTRY_ID, "event": 3, "time": "2026-09-01T10:00:00Z"},
            {"element_in": self.my_squad[9], "element_in_cost": self.pl[self.my_squad[9]]["now_cost"],
             "element_out": 9998, "element_out_cost": 50, "entry": self.ENTRY_ID, "event": 7, "time": "2026-10-01T10:00:00Z"}]

    def entry(self, entry_id):
        return {"id": entry_id, "name": "Test FC", "current_event": self.current_gw}

    def entry_history(self, entry_id):
        cur = []
        for ev in range(1, self.current_gw + 1):
            cur.append({"event": ev, "points": 55, "total_points": 55 * ev, "bank": 5, "value": 1000,
                        "event_transfers": {3: 1, 7: 1, 5: 5}.get(ev, 0), "event_transfers_cost": 0})
        return {"current": cur, "past": [], "chips": [{"name": "wildcard", "time": "x", "event": 5},
                                                      {"name": "bboost", "time": "x", "event": 9}]}

    def entry_transfers(self, entry_id):
        return self.my_transfers

    def entry_picks(self, entry_id, event):
        return {"active_chip": None,
                "entry_history": {"event": event, "bank": 5, "value": 1000},
                "picks": [{"element": e, "position": i + 1, "multiplier": 1, "is_captain": i == 0, "is_vice_captain": i == 1}
                          for i, e in enumerate(self.my_squad)]}
