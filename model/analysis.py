"""Glue between the database and the models, plus report tables."""
from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from db.database import read_table
from model.config import BUDGET_TENTHS, LAST_GW, POS_NAME, ModelParams
from model.optimizer import SquadResult, choose_candidates, optimise_squad
from model.projections import Projections, build_projections


@dataclass
class Context:
    players: pd.DataFrame
    teams: pd.DataFrame
    events: pd.DataFrame
    fixtures: pd.DataFrame
    history: pd.DataFrame
    past: pd.DataFrame
    next_gw: int
    params: ModelParams
    proj: Projections            # base model, next_gw .. GW38
    proj_cons: Projections       # long-term-only model, same GWs

    @property
    def xp_all(self) -> pd.DataFrame:
        return self.proj.matrix()

    @property
    def xp_cons_all(self) -> pd.DataFrame:
        return self.proj_cons.matrix()

    def horizon_gws(self, n: int | None = None) -> list[int]:
        n = n or self.params.horizon
        return [g for g in range(self.next_gw, min(self.next_gw + n, LAST_GW + 1))]

    def weights(self, n: int | None = None) -> dict[int, float]:
        return {g: self.params.gw_decay ** (g - self.next_gw) for g in self.horizon_gws(n)}


def find_next_gw(events: pd.DataFrame) -> int:
    nxt = events[events["is_next"] == 1]
    if len(nxt):
        return int(nxt["id"].iloc[0])
    cur = events[(events["is_current"] == 1) & (events["finished"] == 0)]
    if len(cur):
        return int(cur["id"].iloc[0])
    unfinished = events[events["finished"] == 0]
    return int(unfinished["id"].min()) if len(unfinished) else LAST_GW


def conservative(params: ModelParams) -> ModelParams:
    """Long-term-only view: recent form switched off, priors weighted more."""
    return replace(params, recent_weight=0.0, rate_prior_minutes=params.rate_prior_minutes * 1.5)


def load_context(conn, params: ModelParams | None = None) -> Context:
    params = params or ModelParams()
    players = read_table(conn, "players")
    teams = read_table(conn, "teams")
    events = read_table(conn, "events")
    fixtures = read_table(conn, "fixtures")
    history = read_table(conn, "gw_history")
    past = read_table(conn, "past_seasons")
    if players.empty or events.empty:
        raise SystemExit("Database is empty -- run `python run.py refresh` first.")
    next_gw = find_next_gw(events)
    gws = list(range(next_gw, LAST_GW + 1))
    proj = build_projections(players, teams, fixtures, history, past, gws, params)
    proj_c = build_projections(players, teams, fixtures, history, past, gws, conservative(params))
    return Context(players, teams, events, fixtures, history, past, next_gw, params, proj, proj_c)


# --------------------------------------------------------------------------
# Report tables
# --------------------------------------------------------------------------
def best_squad_from_scratch(ctx: Context, horizon: int | None = None,
                            budget: int = BUDGET_TENTHS, time_limit: float = 60.0) -> SquadResult:
    w = ctx.weights(horizon)
    cand = choose_candidates(ctx.players, ctx.xp_all, w)
    return optimise_squad(cand, ctx.xp_all, w, budget=budget,
                          bench_weight=ctx.params.bench_weight, time_limit=time_limit)


def squad_table(ctx: Context, res: SquadResult, gw: int | None = None) -> pd.DataFrame:
    gw = gw or ctx.next_gw
    lu = res.lineups[gw]
    pl = ctx.players.set_index("id")
    tm = ctx.teams.set_index("id")["short_name"]
    w = ctx.weights()
    rows = []
    order = [(i, "XI") for i in lu.xi] + [(i, f"Bench {n}") for n, i in enumerate(lu.bench, 1)]
    for pid, role in order:
        rows.append({
            "role": role, "pos": POS_NAME[int(pl.at[pid, "position"])], "player": pl.at[pid, "web_name"],
            "team": tm.get(int(pl.at[pid, "team_id"]), "?"), "price": pl.at[pid, "now_cost"] / 10,
            f"xPts GW{gw}": round(float(ctx.xp_all.at[pid, gw]), 2),
            "xPts horizon": round(float(sum(ctx.xp_all.at[pid, g] for g in w)), 1),
            "captain": "C" if pid == lu.captain else ("V" if pid == lu.vice else ""),
        })
    return pd.DataFrame(rows)


def value_table(ctx: Context, min_avg_xmins: float = 45.0) -> pd.DataFrame:
    """Bang-for-buck: projected points per £m, and points above a cheap replacement per extra £m."""
    gws = ctx.horizon_gws()
    long = ctx.proj.long[ctx.proj.long["gw"].isin(gws)]
    agg = long.groupby("player_id").agg(xpts=("xpts", "sum"), xmins=("xmins", "mean"),
                                        fixtures=("n_fix", "sum")).reset_index()
    df = ctx.players.merge(agg, left_on="id", right_on="player_id", how="left").fillna({"xpts": 0.0, "xmins": 0.0})
    tm = ctx.teams.set_index("id")["short_name"]
    df["team"] = df["team_id"].map(tm)
    df["price"] = df["now_cost"] / 10
    df["pos"] = df["position"].map(POS_NAME)
    df["xpts_per_m"] = df["xpts"] / df["price"]
    repl = {}
    for pos, sub in df.groupby("position"):
        floor_price = sub["price"].min()
        cheap = sub[(sub["price"] <= floor_price + 0.5) & (sub["xmins"] >= 20)]
        repl[pos] = (float(cheap["xpts"].max()) if len(cheap) else 0.0, float(floor_price))
    def vorp(r):
        rx, rp = repl[r["position"]]
        dp = r["price"] - rp
        return (r["xpts"] - rx) / dp if dp >= 0.5 else float("nan")
    df["xpts_over_repl_per_m"] = df.apply(vorp, axis=1)
    cols = ["pos", "web_name", "team", "price", "xpts", "xmins", "xpts_per_m", "xpts_over_repl_per_m",
            "selected_pct", "status", "news"]
    out = df[cols].rename(columns={"web_name": "player"})
    out["_ok"] = out["xmins"] >= min_avg_xmins
    return out.sort_values(["pos", "xpts_per_m"], ascending=[True, False]).round(2)
