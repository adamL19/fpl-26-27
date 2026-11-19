#!/usr/bin/env python
"""FPL live dashboard -- command line runner.

    python run.py refresh                       # pull fresh data into data/fpl.sqlite
    python run.py players --search Salah        # find player ids
    python run.py team --entry-id 123456        # snapshot your squad -> data/my_team.json
    python run.py report                        # best squad, value tables, transfers, chips

Run `python run.py <command> -h` for options.  Everything runs from the repo root.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

from db.database import connect, read_table
from model.config import CHIP_LABEL, ModelParams, POS_NAME

ROOT = Path(__file__).parent
DEFAULT_DB = ROOT / "data" / "fpl.sqlite"
DEFAULT_TEAM = ROOT / "data" / "my_team.json"
OUT = ROOT / "output"


def _md(df: pd.DataFrame) -> str:
    df = df.astype(object).where(df.notna(), "-")          # show "-" instead of nan
    try:
        return df.to_markdown(index=False)
    except ImportError:                                       # tabulate not installed
        return df.to_string(index=False)


def cmd_refresh(a) -> None:
    from etl.fpl_api import FPLClient
    from etl.pipeline import refresh
    client = FPLClient(cache_dir=ROOT / "data" / "cache", cache_ttl=a.cache_ttl * 60)
    conn = connect(a.db)
    info = refresh(client, conn, all_players=a.all_players)
    print(f"Done: {info['players']} players, {info['histories']} histories -> {a.db}")


def cmd_players(a) -> None:
    conn = connect(a.db)
    pl = read_table(conn, "players"); tm = read_table(conn, "teams").set_index("id")["short_name"]
    q = pl[pl["web_name"].str.contains(a.search, case=False, na=False)
           | pl["second_name"].str.contains(a.search, case=False, na=False)].copy()
    q["team"] = q["team_id"].map(tm); q["pos"] = q["position"].map(POS_NAME); q["price"] = q["now_cost"] / 10
    print(q[["id", "web_name", "team", "pos", "price", "status"]].to_string(index=False))


def cmd_team(a) -> None:
    from etl.fpl_api import FPLClient
    from etl.my_team import build_my_squad
    from model.analysis import find_next_gw
    conn = connect(a.db)
    pl = read_table(conn, "players"); ev = read_table(conn, "events")
    client = FPLClient(cache_dir=ROOT / "data" / "cache", cache_ttl=0)
    next_gw = find_next_gw(ev)
    sq = build_my_squad(client, a.entry_id, pl, max(next_gw - 1, 1))
    if a.free_transfers is not None:
        sq.free_transfers = a.free_transfers
    if a.bank is not None:
        sq.bank = int(round(a.bank * 10))
    if a.chips_used is not None:
        sq.chips_used = [{"name": c.split("@")[0], "event": int(c.split("@")[1])} for c in a.chips_used]
    sq.save(a.team_file)
    names = pl.set_index("id")["web_name"]
    print(f"Saved {a.team_file}\n  squad: {', '.join(names[i] for i in sq.players)}")
    print(f"  bank £{sq.bank/10:.1f}m | est. free transfers {sq.free_transfers} | "
          f"chips used {sq.chips_used or 'none'}")
    for n in sq.notes:
        print("  note:", n)
    print("Edit the JSON if anything is off (e.g. transfers you've queued but not yet confirmed).")


def _load_my_squad(a):
    from etl.my_team import MySquad
    p = Path(a.team_file)
    if p.exists():
        sq = MySquad.load(p)
        if a.free_transfers is not None:
            sq.free_transfers = a.free_transfers
        if a.bank is not None:
            sq.bank = int(round(a.bank * 10))
        return sq
    return None


def cmd_backtest(a) -> None:
    from model.analysis import find_next_gw
    from model.backtest import backtest, rolling_backtest
    conn = connect(a.db)
    last_done = find_next_gw(read_table(conn, "events")) - 1
    last_cut = a.last_cutoff or last_done - a.k
    first = a.first_cutoff
    if last_cut < first:
        raise SystemExit(f"Not enough played gameweeks yet (have {last_done}); need > {first + a.k}.")
    print(f"Walk-forward backtest: project from data up to GW c, score GW c+1..c+{a.k}, "
          f"c = {first}..{last_cut}\n")
    print(rolling_backtest(conn, first, last_cut, a.k).to_string(index=False))
    print("\nspearman: higher is better (rank agreement).  rmse: lower is better (points error per player).")


def cmd_report(a) -> None:
    from etl.my_team import chips_available
    from model.analysis import (best_squad_from_scratch, load_context, squad_table, value_table)
    from model.chips import compute_chip_gains
    from model.transfers import plan_transfers

    params = ModelParams(horizon=a.horizon)
    conn = connect(a.db)
    ctx = load_context(conn, params)
    OUT.mkdir(exist_ok=True)
    pl = ctx.players.set_index("id")
    lines = [f"# FPL report -- next gameweek: GW{ctx.next_gw}\n"]

    # 1. projections -------------------------------------------------------
    horizon = ctx.horizon_gws()
    xp = ctx.xp_all
    proj = ctx.players[["id", "web_name", "position", "team_id", "now_cost", "status", "news"]].copy()
    for g in horizon:
        proj[f"GW{g}"] = proj["id"].map(xp[g]).round(2)
    proj["horizon_xpts"] = proj["id"].map(xp[horizon].sum(axis=1)).round(1)
    proj.sort_values("horizon_xpts", ascending=False).to_csv(OUT / "projections.csv", index=False)

    # 2. value by position -------------------------------------------------
    val = value_table(ctx)
    val.drop(columns="_ok").to_csv(OUT / "value_by_position.csv", index=False)
    lines.append(f"## Best value by position (next {len(horizon)} GWs, starters only)\n")
    for pos in ("GK", "DEF", "MID", "FWD"):
        top = val[(val["pos"] == pos) & val["_ok"]].head(8)[
            ["player", "team", "price", "xpts", "xpts_per_m", "xpts_over_repl_per_m"]]
        lines.append(f"**{pos}**\n\n{_md(top)}\n")

    # 3. best squad from scratch ------------------------------------------
    print("Solving best 100.0m squad ...")
    best = best_squad_from_scratch(ctx, a.horizon)
    tbl = squad_table(ctx, best)
    tbl.to_csv(OUT / "best_squad.csv", index=False)
    lu = best.lineups[ctx.next_gw]
    lines.append(f"## Best possible squad at £100.0m (Wildcard-style, optimised over {len(horizon)} GWs)\n")
    lines.append(f"Cost £{best.cost/10:.1f}m | formation {lu.formation} | "
                 f"horizon score {best.objective:.1f}\n\n{_md(tbl)}\n")

    # 4. your team: transfers + chips -------------------------------------
    sq = _load_my_squad(a)
    if sq is None:
        lines.append("## Your team\n\nNo team loaded. Run `python run.py team --entry-id <id>` "
                     "(or create data/my_team.json) to get transfer and chip advice.\n")
    else:
        avail = chips_available(sq.chips_used, ctx.next_gw)
        lines.append("## Your team\n")
        lines.append(f"Bank £{sq.bank/10:.1f}m | squad value £{sum(sq.sell_price.values())/10:.1f}m (selling) | "
                     f"free transfers {sq.free_transfers} | chips left: "
                     + "; ".join(f"set {s}: {', '.join(CHIP_LABEL[c] for c in sorted(v)) or 'none'}"
                                 for s, v in avail.items()) + "\n")
        print("Planning transfers ...")
        plan = plan_transfers(ctx.players, ctx.xp_all, ctx.xp_cons_all, ctx.weights(), sq.players,
                              sq.sell_price, sq.bank, sq.free_transfers, ctx.params)
        lines.append("### Transfer recommendation\n")
        lines.append(f"**{plan.reason}**\n")
        if plan.pairs:
            lines.append(_md(pd.DataFrame(plan.pairs)[["out", "in", "price_out", "price_in",
                                                       "out_xpts", "in_xpts", "gain"]]) + "\n")
        opt_rows = [{"transfers": o.n_transfers, "hits": o.hits, "horizon_pts": round(o.objective, 1),
                     "net_after_costs": round(o.net, 1),
                     "long_term_check": None if o.robust_net is None else round(o.robust_net, 1)}
                    for o in plan.options]
        lines.append("Options considered:\n\n" + _md(pd.DataFrame(opt_rows)) + "\n")
        (OUT / "transfers.json").write_text(json.dumps(
            {"reason": plan.reason, "free_transfers": plan.free_transfers, "moves": plan.pairs,
             "options": opt_rows}, indent=2))

        print("Planning chips (this solves a few dozen optimisation problems) ...")
        cg = compute_chip_gains(ctx.players, ctx.xp_all, sq.players, sq.sell_price, sq.bank,
                                ctx.next_gw, ctx.params, avail, quick=a.quick,
                                baseline_ids=plan.chosen.squad)
        tab = cg.table.copy()
        tab.insert(0, "n_fix_total", ctx.proj.long.groupby("gw")["n_fix"].sum().reindex(tab.index).fillna(0).astype(int))
        tab.round(2).to_csv(OUT / "chip_plan.csv")
        lines.append("### Chip plan\n")
        if cg.schedule:
            sched = pd.DataFrame([{"chip": CHIP_LABEL[k.split("@")[0]], "set": k.split("@")[1][3:],
                                   "play in GW": g,
                                   "est. gain (pts)": round(float(cg.table.at[g, k.split("@")[0]]), 1)}
                                  for k, g in sorted(cg.schedule.items(), key=lambda kv: kv[1])])
            lines.append(_md(sched) + f"\n\nTotal estimated gain: {cg.total_gain:.1f} pts. "
                         "Gains are estimates and interact (e.g. a Wildcard changes your squad); "
                         "re-run weekly as doubles/blanks are confirmed.\n")
        else:
            lines.append("No chips left to schedule.\n")
        top_gain = cg.table[["wildcard", "freehit", "bboost", "3xc"]].round(1)
        lines.append("Estimated gain by gameweek (pts):\n\n" + _md(top_gain.reset_index()) + "\n")

    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nFiles written to {OUT}/")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("refresh", help="pull data from the FPL API")
    p.add_argument("--all-players", action="store_true", help="fetch history for every player")
    p.add_argument("--cache-ttl", type=float, default=0, help="reuse cached API responses younger than N minutes")
    p.set_defaults(fn=cmd_refresh)

    p = sub.add_parser("players", help="search player ids")
    p.add_argument("--search", required=True)
    p.set_defaults(fn=cmd_players)

    p = sub.add_parser("team", help="snapshot your squad from your FPL entry id")
    p.add_argument("--entry-id", type=int, required=True)
    p.add_argument("--team-file", default=str(DEFAULT_TEAM))
    p.add_argument("--free-transfers", type=int)
    p.add_argument("--bank", type=float, help="in £m, e.g. 0.5")
    p.add_argument("--chips-used", nargs="*", help="e.g. wildcard@5 bboost@9  (names: wildcard freehit bboost 3xc)")
    p.set_defaults(fn=cmd_team)

    p = sub.add_parser("report", help="projections, best squad, transfers, chip plan")
    p.add_argument("--team-file", default=str(DEFAULT_TEAM))
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--free-transfers", type=int)
    p.add_argument("--bank", type=float)
    p.add_argument("--quick", action="store_true", help="faster, rougher chip scan")
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("backtest", help="check the model against held-out gameweeks")
    p.add_argument("--k", type=int, default=3, help="gameweeks predicted per test")
    p.add_argument("--first-cutoff", type=int, default=6)
    p.add_argument("--last-cutoff", type=int)
    p.set_defaults(fn=cmd_backtest)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
