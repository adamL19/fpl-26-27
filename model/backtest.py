"""Walk-forward backtest: project using only data up to `cutoff`, then compare with
what players actually scored in the following `k` gameweeks.

Works on any populated database (real or synthetic).  Baselines:
  * season_ppg  - points per game so far, extrapolated
  * form_last4  - points per game over the last 4 gameweeks (the "chase the hot hand" approach)
Known simplifications: set-piece orders are today's (mild leakage); injuries are ignored
(everyone treated as fit) because historical status isn't stored.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from db.database import read_table
from model.config import ModelParams
from model.projections import build_projections


def _prepare(conn, cutoff: int, k: int):
    players = read_table(conn, "players"); teams = read_table(conn, "teams")
    fixtures = read_table(conn, "fixtures"); history = read_table(conn, "gw_history")
    past = read_table(conn, "past_seasons")
    hin = history[history["round"] <= cutoff]
    hout = history[(history["round"] > cutoff) & (history["round"] <= cutoff + k)]
    agg = hin.groupby("player_id").agg(
        minutes=("minutes", "sum"), goals=("goals", "sum"), assists=("assists", "sum"),
        xg=("xg", "sum"), xa=("xa", "sum"), bonus=("bonus", "sum"), saves=("saves", "sum"),
        yellow=("yellow", "sum"), red=("red", "sum"))
    p = players.drop(columns=list(agg.columns)).merge(agg, left_on="id", right_index=True, how="left")
    for c in agg.columns:
        p[c] = p[c].fillna(0)
    p["status"] = "a"; p["chance_next"] = np.nan; p["news"] = ""
    fx = fixtures.copy()
    later = fx["event"] > cutoff
    fx.loc[later, "finished"] = 0
    fx.loc[later, ["team_h_score", "team_a_score"]] = np.nan
    return p, teams, fx, hin, hout, past


def backtest(conn, cutoff: int, k: int = 3, params: ModelParams | None = None) -> pd.DataFrame:
    """Returns one row per model/baseline with Spearman rank correlation and RMSE
    versus realised points over gameweeks cutoff+1 .. cutoff+k."""
    params = params or ModelParams()
    p, teams, fx, hin, hout, past = _prepare(conn, cutoff, k)
    gws = list(range(cutoff + 1, cutoff + k + 1))
    actual = hout.groupby("player_id")["total_points"].sum()

    def model(pr: ModelParams) -> pd.Series:
        return build_projections(p, teams, fx, hin, past, gws, pr).matrix().sum(axis=1)

    preds = {
        "model (default)": model(params),
        "model (form off)": model(replace(params, recent_weight=0.0)),
        "model (form 50%)": model(replace(params, recent_weight=0.5, recent_prior_minutes=90)),
    }
    pts = hin.sort_values("round")
    games = pts.groupby("player_id").size()
    preds["season_ppg"] = (pts.groupby("player_id")["total_points"].sum() / games * k)
    last4 = pts.groupby("player_id").tail(4).groupby("player_id")["total_points"].mean()
    preds["form_last4"] = last4 * k

    base = p[p["minutes"] > 270]["id"]                    # players with a meaningful sample
    rows = []
    for name, s in preds.items():
        df = pd.concat([s.rename("pred"), actual.rename("actual")], axis=1).reindex(base).fillna(0.0)
        rows.append({"predictor": name, "players": len(df),
                     "spearman": df["pred"].corr(df["actual"], method="spearman"),
                     "rmse": float(np.sqrt(((df["pred"] - df["actual"]) ** 2).mean())),
                     "mean_pred": df["pred"].mean(), "mean_actual": df["actual"].mean()})
    return pd.DataFrame(rows).round(3)


def rolling_backtest(conn, first_cutoff: int, last_cutoff: int, k: int = 3,
                     params: ModelParams | None = None) -> pd.DataFrame:
    frames = []
    for c in range(first_cutoff, last_cutoff + 1, k):
        df = backtest(conn, c, k, params)
        df.insert(0, "cutoff_gw", c)
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    summary = out.groupby("predictor")[["spearman", "rmse"]].mean().round(3).reset_index()
    return summary.sort_values("spearman", ascending=False)
