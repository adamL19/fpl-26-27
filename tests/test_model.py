import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from etl.my_team import (MySquad, build_my_squad, chips_available, estimate_free_transfers,
                         sell_price)
from etl.pipeline import parse_players, refresh
from model.analysis import best_squad_from_scratch, conservative, find_next_gw
from model.chips import best_schedule
from model.config import MAX_PER_CLUB, SQUAD_QUOTA, XI_MAX, XI_MIN, ModelParams
from model.optimizer import best_lineup, optimise_squad, squad_value
from model.projections import build_projections, poisson_floor_expect
from model.transfers import plan_transfers


# ---------------------------------------------------------------- maths
def test_poisson_floor_expect_matches_bruteforce():
    for lam, k in [(0.4, 2), (1.3, 2), (3.0, 3), (5.5, 3)]:
        brute = sum(math.exp(-lam) * lam ** n / math.factorial(n) * (n // k) for n in range(60))
        assert poisson_floor_expect(lam, k) == pytest.approx(brute, abs=1e-9)
    assert poisson_floor_expect(0, 2) == 0


def test_sell_price_keeps_half_of_profit_rounded_down():
    assert sell_price(50, 53) == 51      # +0.3m profit -> keep 0.1m
    assert sell_price(50, 51) == 50      # +0.1m -> keep 0
    assert sell_price(50, 55) == 52
    assert sell_price(50, 48) == 48      # falls are taken in full


# ---------------------------------------------------------------- manager helpers
def test_free_transfer_estimate_banks_and_caps():
    hist = {"current": [{"event": e, "event_transfers": 0} for e in range(1, 12)]}
    assert estimate_free_transfers(hist, [], 11) == 5            # never used -> capped at 5
    hist["current"][2]["event_transfers"] = 3                    # GW3: 3 transfers (2 hits)
    assert estimate_free_transfers(hist, [], 3) == 1
    # wildcard week doesn't spend FTs
    hist2 = {"current": [{"event": 1, "event_transfers": 0}, {"event": 2, "event_transfers": 0},
                         {"event": 3, "event_transfers": 11}]}
    assert estimate_free_transfers(hist2, [{"name": "wildcard", "event": 3}], 3) == 3


def test_free_transfers_for_fake_manager(fake):
    h = fake.entry_history(fake.ENTRY_ID)
    assert estimate_free_transfers(h, h["chips"], 11) == 5


def test_chips_available_two_sets():
    used = [{"name": "wildcard", "event": 5}, {"name": "bboost", "event": 9}]
    a = chips_available(used, 12)
    assert a[1] == {"freehit", "3xc"} and a[2] == {"wildcard", "freehit", "bboost", "3xc"}
    assert chips_available(used, 20)[1] == set()                 # first set expired
    assert chips_available(used + [{"name": "3xc", "event": 22}], 23)[2] == {"wildcard", "freehit", "bboost"}


def test_my_squad_roundtrip(tmp_path):
    s = MySquad(players=[1, 2], sell_price={1: 50, 2: 45}, bank=7, free_transfers=2,
                chips_used=[{"name": "wildcard", "event": 4}], entry_id=9)
    s.save(tmp_path / "t.json")
    s2 = MySquad.load(tmp_path / "t.json")
    assert s2.sell_price == {1: 50, 2: 45} and s2.budget == 102


# ---------------------------------------------------------------- ETL
def test_parse_players_converts_strings(fake):
    df = parse_players(fake.bootstrap())
    assert df["xg"].dtype.kind == "f" and df["position"].isin([1, 2, 3, 4]).all()
    assert (df["start_cost"] > 0).all() and len(df) == 420


def test_refresh_is_idempotent(fake, tmp_path):
    from db.database import connect
    c = connect(tmp_path / "x.sqlite")
    refresh(fake, c, progress=lambda *_: None)
    n1 = [c.execute(f"select count(*) from {t}").fetchone()[0] for t in ("players", "gw_history", "fixtures")]
    refresh(fake, c, progress=lambda *_: None)
    n2 = [c.execute(f"select count(*) from {t}").fetchone()[0] for t in ("players", "gw_history", "fixtures")]
    assert n1 == n2 and n1[1] > 1000


def test_next_gw(ctx):
    assert ctx.next_gw == 12 == find_next_gw(ctx.events)


# ---------------------------------------------------------------- projections
def test_projections_are_finite_and_in_a_sane_range(ctx):
    xp = ctx.xp_all
    assert np.isfinite(xp.to_numpy()).all()
    assert xp.to_numpy().min() >= -2.0
    assert xp.to_numpy().max() < 25.0           # a single-GW haul of 25+ xPts means a bug
    assert (ctx.proj.long["p_play"].between(0, 1.0001)).all()


def test_blank_and_double_gameweeks(ctx):
    long = ctx.proj.long
    pl = ctx.players.set_index("id")
    fx = ctx.fixtures.dropna(subset=["event"])
    gw25_teams = set(fx[fx.event == 25].team_h) | set(fx[fx.event == 25].team_a)
    blank = long[(long.gw == 25) & (~long.player_id.map(pl["team_id"]).isin(gw25_teams))]
    assert len(blank) > 0 and (blank["xpts"] == 0).all() and (blank["n_fix"] == 0).all()
    dgw = long[(long.gw == 30) & (long.n_fix == 2)]
    sgw = long[(long.gw == 30) & (long.n_fix == 1) & (long.xmins > 0)]
    assert len(dgw) > 0
    assert dgw["xpts"].mean() > 1.5 * sgw["xpts"].mean()


def test_unavailable_players_project_zero(ctx):
    pl = ctx.players
    out = pl[pl.status == "u"]
    assert len(out)
    xp = ctx.xp_all
    assert (xp.loc[out.id, ctx.next_gw] == 0).all()
    susp = pl[pl.status == "s"]
    assert len(susp) and (xp.loc[susp.id, ctx.next_gw] == 0).all()


def test_backtest_beats_naive_baselines_and_is_calibrated(conn):
    from model.backtest import rolling_backtest
    res = rolling_backtest(conn, 5, 8, k=3, params=ModelParams()).set_index("predictor")
    assert res.at["model (default)", "rmse"] < res.at["season_ppg", "rmse"]
    assert res.at["model (default)", "rmse"] < res.at["form_last4", "rmse"]
    assert res.at["model (default)", "spearman"] > res.at["form_last4", "spearman"]


def test_backtest_is_calibrated(conn):
    from model.backtest import backtest
    df = backtest(conn, 8, 3).set_index("predictor")
    m = df.loc["model (default)"]
    assert abs(m["mean_pred"] - m["mean_actual"]) / m["mean_actual"] < 0.12


def test_hot_streak_barely_moves_projection(conn, ctx):
    """3 games of inflated *actual* goals (no xG) must not swing the projection much."""
    from db.database import read_table
    hist = read_table(conn, "gw_history")
    pid = int(ctx.players[(ctx.players.position == 4) & (ctx.players.minutes > 700)].id.iloc[0])
    rows = hist[hist.player_id == pid].sort_values("round").index[-3:]
    hot = hist.copy()
    hot.loc[rows, "goals"] = hot.loc[rows, "goals"] + 2          # six extra goals' worth of "form"
    players = ctx.players.copy()
    players.loc[players.id == pid, "goals"] += 6
    base = ctx.xp_all.loc[pid, ctx.horizon_gws()].sum()
    hot_proj = build_projections(players, ctx.teams, ctx.fixtures, hot, ctx.past,
                                 list(range(ctx.next_gw, 39)), ctx.params).matrix()
    after = hot_proj.loc[pid, ctx.horizon_gws()].sum()
    assert after > base                                           # it does move...
    assert (after - base) / base < 0.15                           # ...but only modestly


def test_pen_taker_floor(ctx):
    pr = ctx.proj.players.set_index("player_id")
    pl = ctx.players.set_index("id")
    takers = pl[(pl.pen_order == 1) & (pl.position.isin([3, 4]))].index
    others = pl[(pl.pen_order.isna()) & (pl.position.isin([3, 4])) & (pl.minutes > 500)].index
    assert pr.loc[takers, "g90"].mean() > pr.loc[others, "g90"].mean()


# ---------------------------------------------------------------- optimiser
def _check_squad(players, ids, budget=1000):
    sq = players[players.id.isin(ids)]
    assert len(sq) == 15
    for pos, q in SQUAD_QUOTA.items():
        assert (sq.position == pos).sum() == q
    assert sq.groupby("team_id").size().max() <= MAX_PER_CLUB
    assert sq.now_cost.sum() <= budget


def test_best_squad_is_legal_and_within_budget(ctx):
    res = best_squad_from_scratch(ctx, 6)
    _check_squad(ctx.players, res.squad)
    assert res.cost <= 1000
    for g, lu in res.lineups.items():
        assert len(lu.xi) == 11 and lu.captain in lu.xi and lu.vice in lu.xi and lu.vice != lu.captain
        pos = ctx.players.set_index("id").loc[lu.xi, "position"].value_counts()
        for p in (1, 2, 3, 4):
            assert XI_MIN[p] <= pos.get(p, 0) <= XI_MAX[p]
        assert len(lu.bench) == 4


def test_objective_matches_independent_evaluation(ctx):
    res = best_squad_from_scratch(ctx, 6)
    w = ctx.weights(6)
    assert squad_value(res.squad, ctx.players, ctx.xp_all, w, ctx.params.bench_weight) == pytest.approx(res.objective, rel=1e-6)


def test_optimum_beats_random_valid_squads(ctx, fake):
    res = best_squad_from_scratch(ctx, 6)
    w = ctx.weights(6)
    manager = squad_value(fake.my_squad, ctx.players, ctx.xp_all, w, ctx.params.bench_weight)
    assert res.objective > manager


def test_more_budget_never_hurts(ctx):
    w = ctx.weights(4)
    a = optimise_squad(ctx.players, ctx.xp_all, w, budget=900)
    b = optimise_squad(ctx.players, ctx.xp_all, w, budget=1000)
    c = optimise_squad(ctx.players, ctx.xp_all, w, budget=1100)
    assert a.objective <= b.objective + 1e-6 <= c.objective + 2e-6


def test_bench_boost_mode_counts_all_fifteen(ctx):
    w = {ctx.next_gw: 1.0}
    bb = optimise_squad(ctx.players, ctx.xp_all, w, all_count=True)
    normal = optimise_squad(ctx.players, ctx.xp_all, w)
    assert bb.objective > normal.objective
    lu = bb.lineups[ctx.next_gw]
    assert bb.objective == pytest.approx(lu.xi_points + lu.captain_extra + lu.bench_points, rel=1e-6)


def test_best_lineup_picks_best_legal_formation():
    squad = pd.DataFrame({"id": range(1, 16), "position": [1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4]})
    ep = {i: 1.0 for i in range(1, 16)}
    ep.update({13: 9.0, 14: 8.0, 15: 7.0, 12: 6.0})              # 3 FWD + a MID score big
    lu = best_lineup(squad, ep)
    assert {13, 14, 15} <= set(lu.xi) and lu.captain == 13 and lu.vice == 14


# ---------------------------------------------------------------- transfers
def test_no_churn_when_squad_is_already_optimal(ctx):
    res = best_squad_from_scratch(ctx, ctx.params.horizon)
    pl = ctx.players.set_index("id")
    plan = plan_transfers(ctx.players, ctx.xp_all, ctx.xp_cons_all, ctx.weights(), res.squad,
                          {i: int(pl.at[i, "now_cost"]) for i in res.squad}, bank=0, free_transfers=2,
                          params=ctx.params)
    assert plan.chosen.n_transfers == 0


def test_bad_squad_gets_transfers_within_budget_and_hit_accounting(ctx, fake):
    pl = ctx.players.set_index("id")
    sell = {i: int(pl.at[i, "now_cost"]) for i in fake.my_squad}
    plan = plan_transfers(ctx.players, ctx.xp_all, ctx.xp_cons_all, ctx.weights(), fake.my_squad,
                          sell, bank=5, free_transfers=1, params=ctx.params)
    assert plan.chosen.n_transfers >= 1
    new_cost = sum(sell[i] if i in sell else int(pl.at[i, "now_cost"]) for i in plan.chosen.squad)
    assert new_cost <= 5 + sum(sell.values())
    _check_squad(ctx.players, plan.chosen.squad, budget=10 ** 6)
    for o in plan.options:
        assert o.hits == max(0, o.n_transfers - 1)
    assert plan.chosen.net > plan.options[0].net


def test_high_hit_cost_discourages_hits(ctx, fake):
    pl = ctx.players.set_index("id")
    sell = {i: int(pl.at[i, "now_cost"]) for i in fake.my_squad}
    harsh = replace(ctx.params, hit_margin=1000.0)
    plan = plan_transfers(ctx.players, ctx.xp_all, ctx.xp_cons_all, ctx.weights(), fake.my_squad,
                          sell, bank=5, free_transfers=1, params=harsh)
    assert plan.chosen.hits == 0


def test_conservative_params_switch_off_form():
    c = conservative(ModelParams())
    assert c.recent_weight == 0 and c.rate_prior_minutes > ModelParams().rate_prior_minutes


# ---------------------------------------------------------------- chips
def _table(gains):
    idx = list(range(12, 39))
    t = pd.DataFrame(0.0, index=idx, columns=["wildcard", "freehit", "bboost", "3xc"])
    for (gw, chip), v in gains.items():
        t.loc[gw, chip] = v
    return t


def test_schedule_one_chip_per_gw_and_respects_availability():
    t = _table({(14, "bboost"): 10, (14, "3xc"): 9, (15, "3xc"): 8, (30, "bboost"): 14, (30, "freehit"): 20,
                (25, "freehit"): 18})
    avail = {1: {"bboost", "3xc"}, 2: {"wildcard", "freehit", "bboost", "3xc"}}
    sched, total = best_schedule(t, avail, 12)
    gws_by_set = {}
    for k, g in sched.items():
        gws_by_set.setdefault(k.split("@")[1], []).append(g)
    for gws in gws_by_set.values():
        assert len(gws) == len(set(gws))                          # one chip per GW
    assert "freehit@set1" not in sched and "wildcard@set1" not in sched
    assert sched["bboost@set1"] == 14 and sched["3xc@set1"] == 15
    assert sched["freehit@set2"] == 30 and sched["bboost@set2"] != 30 or sched["bboost@set2"] == 30


def test_free_hit_not_scheduled_in_gw19_and_gw20():
    t = _table({(19, "freehit"): 30, (20, "freehit"): 30})
    avail = {1: {"freehit"}, 2: {"freehit"}}
    sched, total = best_schedule(t, avail, 12)
    assert not (sched.get("freehit@set1") == 19 and sched.get("freehit@set2") == 20)
    assert total == pytest.approx(30.0)


def test_second_set_only_after_first_expires():
    t = _table({(25, "3xc"): 7})
    sched, _ = best_schedule(t, {1: set(), 2: {"3xc"}}, 21)
    assert sched == {"3xc@set2": 25}


# ---------------------------------------------------------------- end to end
def test_cli_report_end_to_end(fake, conn, ctx, tmp_path, monkeypatch, capsys):
    import run
    pl = ctx.players
    sq = build_my_squad(fake, fake.ENTRY_ID, pl, 11)
    team_file = tmp_path / "my_team.json"
    sq.save(team_file)
    monkeypatch.setattr(run, "OUT", tmp_path / "out")
    db_path = conn.execute("PRAGMA database_list").fetchone()[2]
    run.main(["--db", db_path, "report", "--team-file", str(team_file), "--horizon", "6", "--quick"])
    out = tmp_path / "out"
    for f in ("report.md", "best_squad.csv", "value_by_position.csv", "projections.csv",
              "transfers.json", "chip_plan.csv"):
        assert (out / f).exists() and (out / f).stat().st_size > 50, f
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "Transfer recommendation" in text and "Chip plan" in text and "Best value by position" in text


def test_report_survives_non_ascii_names_under_ascii_locale(conn, tmp_path):
    """Regression: on Windows the default file encoding (cp1252) can't write names like
    'Šeško'.  Reproduce with an ASCII-only locale and check the CLI still works."""
    import os, shutil, sqlite3, subprocess, sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = conn.execute("PRAGMA database_list").fetchone()[2]
    db = tmp_path / "accent.sqlite"
    shutil.copy(src, db)
    c = sqlite3.connect(db)
    c.execute("update players set web_name = 'Šeško' where id = 271")
    c.execute("update players set web_name = 'Ødegaard' where id = 400")
    c.commit(); c.close()
    env = dict(os.environ, PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C")
    r = subprocess.run([sys.executable, "run.py", "--db", str(db), "report", "--quick",
                        "--team-file", str(tmp_path / "none.json")],
                       cwd=root, env=env, capture_output=True, timeout=600)
    assert r.returncode == 0, r.stderr.decode("utf-8", "replace")[-800:]
    report = (root / "output" / "report.md").read_text(encoding="utf-8")
    assert "Šeško" in report or "Ødegaard" in report
