"""Squad selection as a mixed-integer linear program (SciPy's HiGHS solver).

One model covers every decision we need:
  * best 15-man squad from scratch at a budget (Wildcard-style),
  * best squad reachable from your current one with at most t transfers,
  * single-gameweek squads (Free Hit),
  * all-15-count squads (Bench Boost upper bound).

Objective = sum over gameweeks of  w_g * ( XI points + captain extra
            + bench_weight * bench points ).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from model.config import (BUDGET_TENTHS, MAX_PER_CLUB, SQUAD_QUOTA, XI_MAX, XI_MIN)


class _Model:
    """Tiny binary-variable MILP builder."""

    def __init__(self):
        self.cost: list[float] = []
        self.r: list[int] = []; self.c: list[int] = []; self.v: list[float] = []
        self.lo: list[float] = []; self.hi: list[float] = []

    def var(self, obj: float = 0.0) -> int:
        self.cost.append(obj)
        return len(self.cost) - 1

    def con(self, coeffs: dict[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        row = len(self.lo)
        for j, val in coeffs.items():
            self.r.append(row); self.c.append(j); self.v.append(val)
        self.lo.append(lo); self.hi.append(hi)

    def solve(self, time_limit: float = 30.0, gap: float = 1e-4) -> np.ndarray:
        n = len(self.cost)
        A = coo_matrix((self.v, (self.r, self.c)), shape=(len(self.lo), n)).tocsr()
        res = milp(c=-np.array(self.cost),            # maximise
                   constraints=LinearConstraint(A, self.lo, self.hi),
                   integrality=np.ones(n), bounds=Bounds(0, 1),
                   options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
        if res.x is None:
            raise RuntimeError(f"Optimiser found no feasible squad: {res.message}")
        return np.round(res.x)


# --------------------------------------------------------------------------
# Lineups for a fixed squad
# --------------------------------------------------------------------------
_FORMATIONS = [(d, m, f) for d in range(3, 6) for m in range(2, 6) for f in range(1, 4)
               if d + m + f == 10]


@dataclass
class Lineup:
    xi: list[int]
    bench: list[int]            # bench order: GK first, then outfield by xPts
    captain: int
    vice: int
    formation: str
    xi_points: float
    captain_extra: float
    bench_points: float


def best_lineup(squad: pd.DataFrame, ep: dict[int, float]) -> Lineup:
    """squad needs columns id, position.  ep maps id -> xPts for the gameweek."""
    by_pos = {pos: sorted(squad.loc[squad["position"] == pos, "id"], key=lambda i: -ep.get(i, 0.0))
              for pos in (1, 2, 3, 4)}
    best = None
    for d, m, f in _FORMATIONS:
        if len(by_pos[2]) < d or len(by_pos[3]) < m or len(by_pos[4]) < f or not by_pos[1]:
            continue
        xi = by_pos[1][:1] + by_pos[2][:d] + by_pos[3][:m] + by_pos[4][:f]
        tot = sum(ep.get(i, 0.0) for i in xi)
        if best is None or tot > best[0]:
            best = (tot, xi, f"{d}-{m}-{f}")
    if best is None:
        raise ValueError("Squad cannot form a legal XI")
    tot, xi, form = best
    ranked = sorted(xi, key=lambda i: -ep.get(i, 0.0))
    bench_gk = [i for i in by_pos[1] if i not in xi]
    bench_out = sorted([i for i in squad["id"] if i not in xi and i not in bench_gk],
                       key=lambda i: -ep.get(i, 0.0))
    bench = bench_gk + bench_out
    return Lineup(xi=xi, bench=bench, captain=ranked[0], vice=ranked[1] if len(ranked) > 1 else ranked[0],
                  formation=form, xi_points=tot, captain_extra=ep.get(ranked[0], 0.0),
                  bench_points=sum(ep.get(i, 0.0) for i in bench))


def squad_value(squad_ids: list[int], players: pd.DataFrame, xp: pd.DataFrame,
                weights: dict[int, float], bench_weight: float) -> float:
    """Same objective the MILP optimises, for an arbitrary fixed squad."""
    squad = players[players["id"].isin(squad_ids)][["id", "position"]]
    total = 0.0
    for g, w in weights.items():
        ep = xp[g].to_dict()
        lu = best_lineup(squad, ep)
        total += w * (lu.xi_points + lu.captain_extra + bench_weight * lu.bench_points)
    return total


# --------------------------------------------------------------------------
# Candidate pruning
# --------------------------------------------------------------------------
def choose_candidates(players: pd.DataFrame, xp: pd.DataFrame, weights: dict[int, float],
                      keep: set[int] | None = None, per_pos: dict[int, int] | None = None,
                      budget_cap_cost: int = 45) -> pd.DataFrame:
    """Keep the most promising players per position (by weighted xPts and by
    points-per-cost, plus cheap bench enablers) so the MILP stays small."""
    per_pos = per_pos or {1: 14, 2: 45, 3: 45, 4: 28}
    df = players.copy()
    df["_w"] = [sum(xp.at[i, g] * w for g, w in weights.items()) if i in xp.index else 0.0
                for i in df["id"]]
    keep_ids = set(keep or [])
    chosen = set(keep_ids)
    for pos, n in per_pos.items():
        sub = df[df["position"] == pos]
        chosen |= set(sub.nlargest(n, "_w")["id"])
        sub = sub.assign(_v=sub["_w"] / sub["now_cost"].clip(lower=38))
        chosen |= set(sub.nlargest(max(6, n // 3), "_v")["id"])
        cheap = sub[sub["now_cost"] <= budget_cap_cost]
        chosen |= set(cheap.nlargest(max(5, n // 4), "_w")["id"])
    return df[df["id"].isin(chosen)].drop(columns="_w").reset_index(drop=True)


# --------------------------------------------------------------------------
# The optimiser
# --------------------------------------------------------------------------
@dataclass
class SquadResult:
    squad: list[int]
    objective: float
    cost: int                       # in budget terms (sell prices for retained players)
    lineups: dict[int, Lineup] = field(default_factory=dict)
    transfers_in: list[int] = field(default_factory=list)
    transfers_out: list[int] = field(default_factory=list)


def optimise_squad(players: pd.DataFrame, xp: pd.DataFrame, weights: dict[int, float],
                   budget: int = BUDGET_TENTHS, bench_weight: float = 0.10,
                   current: list[int] | None = None, sell_price: dict[int, int] | None = None,
                   max_transfers: int | None = None, all_count: bool = False,
                   exclude: set[int] | None = None, include: set[int] | None = None,
                   time_limit: float = 30.0) -> SquadResult:
    """players: id, position, team_id, now_cost.  xp: index player id, columns gw.

    If `current` is given, retained players cost their sell price and new
    players cost now_cost, so `budget` should be bank + sum(sell prices).
    all_count=True makes every squad member score fully (Bench Boost).
    """
    exclude = exclude or set(); include = include or set()
    current_set = set(current or [])
    sell_price = sell_price or {}
    df = players[~players["id"].isin(exclude - include)].reset_index(drop=True)
    gws = list(weights)
    m = _Model()

    s = {}; x = {}; c = {}
    cost_of = {}
    for row in df.itertuples():
        pid = int(row.id)
        cost_of[pid] = int(sell_price.get(pid, row.now_cost)) if pid in current_set else int(row.now_cost)
        # small tie-break toward spending less
        s[pid] = m.var(-1e-5 * cost_of[pid] / 10)
        for g in gws:
            ep = float(xp.at[pid, g]) if pid in xp.index else 0.0
            w = weights[g]
            if all_count:
                x[pid, g] = m.var(0.0)
                c[pid, g] = m.var(w * ep)
                m.cost[s[pid]] += w * ep           # every squad member scores once
            else:
                x[pid, g] = m.var(w * ep * (1 - bench_weight))
                c[pid, g] = m.var(w * ep)
                m.cost[s[pid]] += w * ep * bench_weight
    ids = list(df["id"].astype(int))
    pos_of = dict(zip(df["id"].astype(int), df["position"].astype(int)))
    team_of = dict(zip(df["id"].astype(int), df["team_id"].astype(int)))

    m.con({s[i]: cost_of[i] for i in ids}, hi=budget)
    m.con({s[i]: 1 for i in ids}, lo=15, hi=15)
    for pos, q in SQUAD_QUOTA.items():
        m.con({s[i]: 1 for i in ids if pos_of[i] == pos}, lo=q, hi=q)
    for t in set(team_of.values()):
        m.con({s[i]: 1 for i in ids if team_of[i] == t}, hi=MAX_PER_CLUB)
    for i in include:
        if i in s:
            m.con({s[i]: 1}, lo=1)
    for g in gws:
        m.con({x[i, g]: 1 for i in ids}, lo=11, hi=11)
        for pos in (1, 2, 3, 4):
            m.con({x[i, g]: 1 for i in ids if pos_of[i] == pos}, lo=XI_MIN[pos], hi=XI_MAX[pos])
        m.con({c[i, g]: 1 for i in ids}, lo=1, hi=1)
        for i in ids:
            m.con({x[i, g]: 1, s[i]: -1}, hi=0)
            m.con({c[i, g]: 1, x[i, g]: -1}, hi=0)
    if current_set and max_transfers is not None:
        new = [i for i in ids if i not in current_set]
        m.con({s[i]: 1 for i in new}, hi=max_transfers)

    sol = m.solve(time_limit=time_limit)
    squad = [i for i in ids if sol[s[i]] > 0.5]
    sq_df = df[df["id"].isin(squad)][["id", "position"]]
    lineups = {g: best_lineup(sq_df, xp[g].to_dict()) for g in gws}
    obj = sum(weights[g] * (lu.xi_points + lu.captain_extra + (1.0 if all_count else bench_weight) * lu.bench_points)
              for g, lu in lineups.items())
    return SquadResult(
        squad=squad, objective=obj, cost=sum(cost_of[i] for i in squad), lineups=lineups,
        transfers_in=[i for i in squad if current_set and i not in current_set],
        transfers_out=[i for i in current_set if i not in squad] if current_set else [])
