"""Chip timing: when are Wildcard, Free Hit, Bench Boost and Triple Captain worth most?

Rules (2026/27): two sets of four chips; set 1 must be used before the GW19
deadline, set 2 opens at GW20; one chip per gameweek; Free Hit can't be played in
consecutive weeks across the GW19/GW20 boundary.

For every candidate gameweek we estimate each chip's *gain in points* and then
pick the best non-conflicting schedule per set.

How each gain is estimated
  Triple Captain  extra points of the best captain (1x his xPts). Near-term uses
                  your squad; further out uses the best captain in the game x 0.85
                  (you can plan to own him).
  Bench Boost     xPts of your 4 bench players (near-term) or of the bench of the
                  best "bench-boost-built" squad x 0.7 (further out).
  Free Hit        best possible single-gameweek squad minus what your regular
                  squad would score (near-term: your actual squad; further out: a
                  squad built for the surrounding gameweeks, which isolates the
                  blank/double-gameweek advantage).
  Wildcard        points gained by a fresh squad over the next few weeks versus
                  holding; further out it is scaled by 0.5 because you will have
                  used normal transfers by then.

These are estimates -- fixtures for double/blank gameweeks are only known once the
league announces rearrangements, so re-run weekly.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import pandas as pd

from model.config import (CHIP_LABEL, FIRST_HALF_LAST_GW, LAST_GW,
                          SECOND_HALF_FIRST_GW, ModelParams)
from model.optimizer import (best_lineup, choose_candidates, optimise_squad, squad_value)

NEAR_WINDOW = 4          # GWs where "your squad" is trusted as-is
WC_LOOKAHEAD = 6


@dataclass
class ChipGains:
    table: pd.DataFrame        # index gw; columns wildcard, freehit, bboost, 3xc (+ detail columns)
    schedule: dict[str, int]   # chip -> gw (best schedule over available chips)
    total_gain: float


def _single(gw: int) -> dict[int, float]:
    return {gw: 1.0}


def _window(center: int, lo: int, hi: int, n: int, exclude: int | None = None) -> dict[int, float]:
    gws = [g for g in range(max(lo, center - n // 2), min(hi, center + n // 2) + 1) if g != exclude]
    return {g: 1.0 for g in gws}


def compute_chip_gains(players: pd.DataFrame, xp_all: pd.DataFrame, squad_ids: list[int],
                       sell_price: dict[int, int], bank: int, next_gw: int,
                       params: ModelParams, available: dict[int, set[str]],
                       time_limit: float = 15.0, quick: bool = False,
                       baseline_ids: list[int] | None = None) -> ChipGains:
    """xp_all must cover every GW from next_gw to the end of the season.

    squad_ids / sell_price / bank describe what you own today (they set the budget).
    baseline_ids is the squad chip gains are measured against -- pass the squad you
    would have after this week's recommended transfers so normal transfers aren't
    credited to the chips.  Defaults to squad_ids."""
    budget = bank + sum(sell_price.values())
    base_ids = list(baseline_ids or squad_ids)
    keep = set(squad_ids) | set(base_ids)
    gws = [g for g in xp_all.columns if g >= next_gw]
    sq = players[players["id"].isin(base_ids)][["id", "position"]]
    rows = []
    near_hi = next_gw + NEAR_WINDOW - 1

    for g in gws:
        ep = xp_all[g].to_dict()
        lu = best_lineup(sq, ep)
        near = g <= near_hi
        row = {"gw": g, "your_xi": lu.xi_points + lu.captain_extra,
               "tc_your": lu.captain_extra, "bb_your": lu.bench_points}
        # --- ideal references (single-GW) --------------------------------
        top = xp_all[g].nlargest(1)
        row["tc_ideal"] = float(top.iloc[0]) if len(top) else 0.0
        w1 = _single(g)
        cand = choose_candidates(players, xp_all, w1, keep=keep)
        fh = optimise_squad(cand, xp_all, w1, budget=budget, bench_weight=params.bench_weight,
                            time_limit=time_limit)
        row["fh_best"] = fh.objective
        if not quick or near:
            bb = optimise_squad(cand, xp_all, w1, budget=budget, all_count=True, time_limit=time_limit)
            lu_bb = bb.lineups[g]
            row["bb_ideal"] = lu_bb.bench_points
        else:
            row["bb_ideal"] = float("nan")
        # --- free hit baseline -------------------------------------------
        if near:
            base = row["your_xi"]
        else:
            win = _window(g, next_gw, LAST_GW, 6, exclude=g)
            base_squad = optimise_squad(choose_candidates(players, xp_all, win, keep=keep),
                                        xp_all, win, budget=budget,
                                        bench_weight=params.bench_weight, time_limit=time_limit)
            lu_b = best_lineup(players[players["id"].isin(base_squad.squad)][["id", "position"]], ep)
            base = lu_b.xi_points + lu_b.captain_extra
        row["fh_baseline"] = base
        row["fh_gain"] = max(0.0, fh.objective - base)
        # --- planning gains ----------------------------------------------
        row["3xc"] = row["tc_your"] if near else 0.85 * max(row["tc_ideal"], row["tc_your"])
        bb_far = 0.7 * row["bb_ideal"] if row["bb_ideal"] == row["bb_ideal"] else row["bb_your"]
        row["bboost"] = row["bb_your"] if near else max(bb_far, row["bb_your"])
        row["freehit"] = row["fh_gain"]
        row["wildcard"] = 0.0
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("gw")

    # --- wildcard: only a subset of gameweeks (each needs two multi-GW solves) ----
    wc_gws = [g for g in gws if g <= next_gw + 3 or g == FIRST_HALF_LAST_GW
              or (g >= SECOND_HALF_FIRST_GW and (g - SECOND_HALF_FIRST_GW) % 3 == 0)]
    if quick:
        wc_gws = [g for g in wc_gws if g <= next_gw + 1 or g in (FIRST_HALF_LAST_GW, SECOND_HALF_FIRST_GW)]
    ideal_now = None
    for g in wc_gws:
        wts = {k: params.gw_decay ** (k - g) for k in range(g, min(g + WC_LOOKAHEAD, LAST_GW + 1))}
        cand = choose_candidates(players, xp_all, wts, keep=keep)
        fresh = optimise_squad(cand, xp_all, wts, budget=budget, bench_weight=params.bench_weight,
                               time_limit=time_limit)
        if g <= near_hi:
            hold = squad_value(base_ids, players, xp_all, wts, params.bench_weight)
            scale = 1.0
        else:
            if ideal_now is None:
                w_now = {k: params.gw_decay ** (k - next_gw) for k in range(next_gw, next_gw + WC_LOOKAHEAD)}
                ideal_now = optimise_squad(choose_candidates(players, xp_all, w_now, keep=keep),
                                           xp_all, w_now, budget=budget,
                                           bench_weight=params.bench_weight, time_limit=time_limit).squad
            hold = squad_value(ideal_now, players, xp_all, wts, params.bench_weight)
            scale = 0.5
        tbl.loc[g, "wildcard"] = max(0.0, scale * (fresh.objective - hold))

    schedule, total = best_schedule(tbl, available, next_gw)
    return ChipGains(table=tbl, schedule=schedule, total_gain=total)


def best_schedule(tbl: pd.DataFrame, available: dict[int, set[str]], next_gw: int):
    """Exhaustive search for the best assignment of remaining chips to distinct
    gameweeks, per set, honouring the no-consecutive-Free-Hit rule across the break."""
    best_by_set: dict[int, tuple[float, dict[str, int]]] = {}
    for s, (lo, hi) in {1: (next_gw, FIRST_HALF_LAST_GW), 2: (SECOND_HALF_FIRST_GW, LAST_GW)}.items():
        chips = sorted(available.get(s, set()))
        gws = [g for g in tbl.index if lo <= g <= hi]
        top = (0.0, {})
        if not chips or not gws:
            best_by_set[s] = top
            continue
        # only the most valuable few GWs per chip can matter; prune for speed
        cand_gws = sorted({g for c in chips for g in tbl[c].loc[gws].nlargest(6).index})
        for combo in itertools.permutations(cand_gws, min(len(chips), len(cand_gws))):
            assign = dict(zip(chips, combo))
            val = sum(float(tbl.at[g, c]) for c, g in assign.items())
            if val > top[0]:
                top = (val, assign)
        best_by_set[s] = top
    # Free Hit cannot be used in GW19 and GW20
    (v1, a1), (v2, a2) = best_by_set[1], best_by_set[2]
    if a1.get("freehit") == FIRST_HALF_LAST_GW and a2.get("freehit") == SECOND_HALF_FIRST_GW:
        alt2 = _reassign_without(tbl, available.get(2, set()), SECOND_HALF_FIRST_GW, "freehit")
        alt1 = _reassign_without(tbl, available.get(1, set()), FIRST_HALF_LAST_GW, "freehit", lo=next_gw, hi=FIRST_HALF_LAST_GW)
        if alt2[0] + v1 >= alt1[0] + v2:
            v2, a2 = alt2
        else:
            v1, a1 = alt1
    sched = {}
    for s, a in ((1, a1), (2, a2)):
        for c, g in a.items():
            sched[f"{c}@set{s}"] = int(g)
    return sched, v1 + v2


def _reassign_without(tbl, chips, banned_gw, banned_chip, lo=SECOND_HALF_FIRST_GW, hi=LAST_GW):
    chips = sorted(chips)
    gws = [g for g in tbl.index if lo <= g <= hi]
    top = (0.0, {})
    cand_gws = sorted({g for c in chips for g in tbl[c].loc[gws].nlargest(6).index} | set())
    for combo in itertools.permutations(cand_gws, min(len(chips), len(cand_gws))):
        assign = dict(zip(chips, combo))
        if assign.get(banned_chip) == banned_gw:
            continue
        val = sum(float(tbl.at[g, c]) for c, g in assign.items())
        if val > top[0]:
            top = (val, assign)
    return top
