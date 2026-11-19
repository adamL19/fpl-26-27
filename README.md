# FPL Live Dashboard — data pipeline, projections, squad & chip optimiser

Pulls Fantasy Premier League data, projects expected points for every player over the
coming gameweeks, and uses an optimiser to recommend the best £100.0m squad, this
week's transfers, and when to play Wildcard / Free Hit / Bench Boost / Triple Captain.

> **Status:** the code was built and tested against a *synthetic* league that mimics the
> FPL API (the live API wasn't reachable from the build environment). Run
> `python scripts/check_schema.py` first — it verifies the real API still has every field
> the code needs and prints sanity checks. Then run the backtest on your real data.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/check_schema.py                         # 1. verify the live API schema
python run.py refresh                                  # 2. pull data (~1-2 min first time)
python run.py players --search Salah                   #    (find player ids if needed)
python run.py team --entry-id 1234567                  # 3. snapshot YOUR team (id is in your FPL URL)
python run.py report                                   # 4. projections, best squad, transfers, chips
python run.py backtest                                 # 5. how good is the model on real GWs so far?
python -m pytest -q                                    #    run the test suite (offline)
```

Outputs land in `output/`: `report.md` (start here), `best_squad.csv`,
`value_by_position.csv` (bang-for-buck), `projections.csv`, `transfers.json`, `chip_plan.csv`.

### About your team snapshot
`run.py team` reads your public FPL data. Two things it **cannot** see, so check
`data/my_team.json` and edit it if needed:
* transfers you've queued for the upcoming gameweek (it shows your squad as of the last deadline);
* your exact free-transfer count (estimated by replaying your history). Override with
  `--free-transfers N`, bank with `--bank 0.5`, chips with `--chips-used wildcard@5 bboost@9`.

Selling prices are rebuilt from your transfer history (half of any price rise, rounded down).

## How it works

| Stage | File | What it does |
|---|---|---|
| Extract | `etl/fpl_api.py`, `etl/pipeline.py` | Public API → SQLite (`db/schema.sql`): players, teams, fixtures, per-GW history, past seasons, set-piece orders, defensive contributions |
| Team strength | `model/team_ratings.py` | Poisson attack/defence ratings from results, shrunk toward FPL's own strength numbers early in the season |
| Projections | `model/projections.py` | xPts per player per GW from goals, assists, minutes, clean sheets, goals conceded, saves, bonus, cards, defensive contribution, penalty/FK/corner takers, fixture difficulty, blanks and doubles |
| Optimiser | `model/optimizer.py` | Mixed-integer program (SciPy HiGHS): legal 15-man squad, budget, ≤3 per club, best XI + captain every GW |
| Transfers | `model/transfers.py` | Best squad for 0–3 transfers, scored net of hits and churn |
| Chips | `model/chips.py` | Gain estimate per chip per GW, best schedule per chip set |
| Validation | `model/backtest.py` | Walk-forward test vs. naive "season average" and "last-4 games form" baselines |

### Not being swayed by short-term streaks
* Goal/assist rates use season-long **xG/xA** (75%) over actual goals, shrunk toward a positional + last-season prior by minutes played.
* Last-6-GW form can move a rate by **at most 15%**, and is itself shrunk.
* A transfer must clear a churn cost (1.0 pt each), a banked-transfer value (1.5 pts), and a −4 hit plus 2 pts margin if it uses a hit.
* Every proposed transfer is re-scored under a **long-term-only** model (form switched off); if it only works because of a hot streak, the plan holds instead.
* All of these live in `ModelParams` in `model/config.py` — tune them with `run.py backtest`.

### Rules encoded (2026/27)
Two chip sets (Wildcard, Free Hit, Bench Boost, Triple Captain each). Set 1 must be used
before the **GW19 deadline (Sat 2 Jan 2027)**; set 2 opens at GW20 (you said "week 19 or
January 1st" — that's the same boundary). One chip per gameweek; no Free Hit in GW19 *and*
GW20. Up to 5 banked free transfers; hits cost 4. £100.0m budget, 2/5/5/3 squad, ≤3 per club.

## Known limitations — please read
1. **Defensive-contribution thresholds** (10 for defenders, 12 for mids/forwards) are the 2025/26 values; the announcement for 2026/27 didn't state new ones. `check_schema.py` prints the live hit-rates so you can sanity-check. Edit `DC_THRESHOLD` in `model/config.py` if they changed.
2. **Chip gains are estimates.** Doubles/blanks are only known once fixtures are rearranged, chips interact with each other, and far-future gains use rougher proxies (documented in `model/chips.py`). Re-run weekly.
3. **Injury returns are assumptions** (unknown-length injuries heal linearly over 6 GWs; suspensions 2 GWs). See `ModelParams`.
4. **Fixture model is simple** (team Poisson ratings, no lineup/xG-by-opponent detail, no cup congestion or manager-rotation signals).
5. **Not validated on real data yet.** On synthetic data the model was well calibrated and beat naive baselines on error, but the synthetic league has constant talent by construction, so it says nothing about whether real-world form carries signal. Run `python run.py backtest` on live data and tune `recent_weight` accordingly.
6. Unofficial API: be polite (the client throttles requests), and expect it to change without notice.

## Layout
```
etl/        API client, parsers, your-team snapshot
db/         schema.sql + SQLite helpers
model/      ratings, projections, optimiser, transfers, chips, backtest, config
tests/      30 offline tests + a synthetic-league generator
scripts/    check_schema.py
app/        (reserved for the Streamlit/Dash dashboard — next step)
data/ output/   generated, git-ignored
```

## Next steps
* Streamlit dashboard in `app/` reading `data/fpl.sqlite` + `output/`, auto-refreshing.
* Scheduled refresh (GitHub Actions cron + hosted Postgres) for true "live" updates.
* Tune parameters from the real-data backtest; add price-change and ownership features.
