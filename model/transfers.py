"""Transfer planning with built-in resistance to chasing short-term form.

For each candidate number of transfers t (0..T) we find the best squad reachable
with at most t changes, then score it as

    net(t) = horizon points
             - 4 * hits
             - switch_cost * t                 (churn / model-error penalty)
             + roll_value * extra FTs banked   (flexibility of keeping a transfer)

A plan must also survive a second opinion: the same squads are re-scored under a
"long-term only" projection (recent form switched off).  If a move only looks good
because of a hot streak it is rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from model.config import HIT_COST, MAX_BANKED_FT, ModelParams
from model.optimizer import (choose_candidates, optimise_squad, squad_value)


@dataclass
class TransferOption:
    n_transfers: int
    hits: int
    objective: float
    net: float
    robust_net: float | None
    out: list[int]
    inn: list[int]
    squad: list[int]


@dataclass
class TransferPlan:
    free_transfers: int
    options: list[TransferOption]
    chosen: TransferOption
    baseline_objective: float
    reason: str
    pairs: list[dict] = field(default_factory=list)


def _bank_bonus(ft: int, t: int, roll_value: float) -> float:
    if t > ft:
        return 0.0
    return roll_value * (min(MAX_BANKED_FT, ft - t + 1) - 1)


def _net(obj: float, ft: int, t: int, params: ModelParams) -> float:
    hits = max(0, t - ft)
    penalty = HIT_COST * hits + (params.hit_margin * hits) + params.switch_cost * t
    return obj - penalty + _bank_bonus(ft, t, params.roll_value)


def _pair_moves(out_ids, in_ids, players: pd.DataFrame, xp: pd.DataFrame,
                weights: dict[int, float]) -> list[dict]:
    pl = players.set_index("id")
    outs = sorted(out_ids, key=lambda i: (pl.at[i, "position"], -pl.at[i, "now_cost"]))
    ins = sorted(in_ids, key=lambda i: (pl.at[i, "position"], -pl.at[i, "now_cost"]))
    wsum = lambda i: float(sum(xp.at[i, g] * w for g, w in weights.items()))
    pairs = []
    for o, i in zip(outs, ins):
        pairs.append({"out": pl.at[o, "web_name"], "in": pl.at[i, "web_name"],
                      "out_id": int(o), "in_id": int(i),
                      "out_xpts": round(wsum(o), 2), "in_xpts": round(wsum(i), 2),
                      "gain": round(wsum(i) - wsum(o), 2),
                      "price_out": float(pl.at[o, "now_cost"]) / 10, "price_in": float(pl.at[i, "now_cost"]) / 10})
    return pairs


def plan_transfers(players: pd.DataFrame, xp: pd.DataFrame, xp_conservative: pd.DataFrame,
                   weights: dict[int, float], squad_ids: list[int], sell_price: dict[int, int],
                   bank: int, free_transfers: int, params: ModelParams,
                   time_limit: float = 30.0) -> TransferPlan:
    budget = bank + sum(sell_price.values())
    cand = choose_candidates(players, xp, weights, keep=set(squad_ids))
    base_val = squad_value(squad_ids, players, xp, weights, params.bench_weight)
    base_cons = squad_value(squad_ids, players, xp_conservative, weights, params.bench_weight)

    options: list[TransferOption] = []
    for t in range(0, params.max_transfers_considered + 1):
        if t == 0:
            res_squad, obj, out, inn = list(squad_ids), base_val, [], []
        else:
            r = optimise_squad(cand, xp, weights, budget=budget, bench_weight=params.bench_weight,
                               current=squad_ids, sell_price=sell_price, max_transfers=t,
                               time_limit=time_limit)
            res_squad, obj, out, inn = r.squad, r.objective, r.transfers_out, r.transfers_in
        n_actual = len(inn)
        cons = squad_value(res_squad, players, xp_conservative, weights, params.bench_weight)
        robust = (cons - base_cons) - (params.switch_cost * n_actual)
        options.append(TransferOption(
            n_transfers=n_actual, hits=max(0, n_actual - free_transfers), objective=obj,
            net=_net(obj, free_transfers, n_actual, params), robust_net=robust,
            out=sorted(out), inn=sorted(inn), squad=sorted(res_squad)))

    # unique by n_transfers, keep the best objective for each
    best_by_n: dict[int, TransferOption] = {}
    for o in options:
        if o.n_transfers not in best_by_n or o.objective > best_by_n[o.n_transfers].objective:
            best_by_n[o.n_transfers] = o
    opts = [best_by_n[k] for k in sorted(best_by_n)]

    roll = best_by_n[0]
    ok = [o for o in opts if o.n_transfers > 0
          and o.net > roll.net
          and (o.robust_net is not None and o.robust_net > params.min_robust_gain)]
    if ok:
        chosen = max(ok, key=lambda o: o.net)
        reason = (f"{chosen.n_transfers} transfer(s) beat rolling by {chosen.net - roll.net:.1f} pts "
                  f"after hit/churn costs, and still help under the long-term-only model "
                  f"(+{chosen.robust_net:.1f}).")
    else:
        chosen = roll
        better = [o for o in opts if o.n_transfers > 0 and o.net > roll.net]
        if better:
            reason = ("Some moves look good on recent form but don't clear the long-term-only "
                      "check -- holding rather than chasing form.")
        else:
            reason = "No transfer clears the churn/hit threshold over the horizon -- roll the transfer."
    plan = TransferPlan(free_transfers=free_transfers, options=opts, chosen=chosen,
                        baseline_objective=base_val, reason=reason)
    plan.pairs = _pair_moves(chosen.out, chosen.inn, players, xp, weights) if chosen.n_transfers else []
    return plan
