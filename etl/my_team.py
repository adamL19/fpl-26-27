"""Build a snapshot of *your* team from the public manager endpoints.

Limitations (the authenticated "my-team" endpoint is not used):
  * the squad is as of the last deadline -- transfers you've queued for the
    upcoming gameweek are invisible.  Use data/my_team.json to override.
  * free transfers are *estimated* by replaying your transfer history.
  * selling prices are rebuilt from your transfer history (FPL keeps only half
    of any price rise, rounded down).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from model.config import FIRST_HALF_LAST_GW, MAX_BANKED_FT


@dataclass
class MySquad:
    players: list[int]
    sell_price: dict[int, int]
    bank: int                         # tenths
    free_transfers: int
    chips_used: list[dict] = field(default_factory=list)   # [{"name": "wildcard", "event": 5}]
    entry_id: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def budget(self) -> int:
        return self.bank + sum(self.sell_price.values())

    def save(self, path: str | Path) -> None:
        d = asdict(self)
        d["sell_price"] = {str(k): v for k, v in self.sell_price.items()}
        Path(path).write_text(json.dumps(d, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "MySquad":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["sell_price"] = {int(k): v for k, v in d["sell_price"].items()}
        return MySquad(**d)


def sell_price(buy: int, now: int) -> int:
    """FPL selling price: keep half of any profit, rounded down (in tenths)."""
    return now if now <= buy else buy + (now - buy) // 2


def estimate_free_transfers(history: dict, chips: list[dict], last_event: int) -> int:
    """Replay weekly transfer counts to estimate free transfers for event last_event+1.

    Rules: 1 FT arrives each week, unused ones bank up to 5, a hit is charged for
    transfers beyond the FTs you have, and Wildcard / Free Hit weeks don't consume FTs.
    """
    made = {int(r["event"]): int(r.get("event_transfers", 0)) for r in history.get("current", [])}
    chip_by_event = {int(c["event"]): c["name"] for c in chips}
    ft = 1                                   # entering GW2 (GW1 transfers are unlimited)
    for ev in range(2, last_event + 1):
        used = made.get(ev, 0)
        if chip_by_event.get(ev) in ("wildcard", "freehit"):
            ft = min(MAX_BANKED_FT, ft + 1)
        else:
            ft = min(MAX_BANKED_FT, max(ft - used, 0) + 1)
    return ft


def build_my_squad(client, entry_id: int, players_df, current_event: int) -> MySquad:
    """players_df needs id, now_cost, start_cost."""
    notes: list[str] = []
    history = client.entry_history(entry_id)
    chips = history.get("chips", [])
    chip_by_event = {int(c["event"]): c["name"] for c in chips}

    # Use the latest GW whose squad is a *real* squad (not a one-week Free Hit).
    ev = current_event
    picks = client.entry_picks(entry_id, ev)
    while picks.get("active_chip") == "freehit" and ev > 1:
        ev -= 1
        picks = client.entry_picks(entry_id, ev)
    if ev != current_event:
        notes.append(f"GW{current_event} was a Free Hit; using your GW{ev} squad.")
    squad_ids = [int(p["element"]) for p in picks["picks"]]
    bank = int(picks.get("entry_history", {}).get("bank", 0))

    # Rebuild buy prices from transfer history (ignoring Free Hit weeks).
    transfers = [t for t in client.entry_transfers(entry_id)
                 if chip_by_event.get(int(t["event"])) != "freehit"]
    transfers.sort(key=lambda t: (t["event"], t.get("time", "")))
    last_buy: dict[int, int] = {}
    for t in transfers:
        last_buy[int(t["element_in"])] = int(t["element_in_cost"])
    pl = players_df.set_index("id")
    sell = {}
    n_start = 0
    for pid in squad_ids:
        now = int(pl.at[pid, "now_cost"])
        buy = last_buy.get(pid, int(pl.at[pid, "start_cost"]))
        n_start += pid not in last_buy
        sell[pid] = sell_price(buy, now)
    notes.append(f"{n_start} players have no transfer record (original squad) and are assumed "
                 "bought at their start-of-season price.")

    ft = estimate_free_transfers(history, chips, ev)
    notes.append("Free transfers are an estimate from your transfer history; "
                 "pass --free-transfers to override.")
    return MySquad(players=squad_ids, sell_price=sell, bank=bank, free_transfers=ft,
                   chips_used=[{"name": c["name"], "event": int(c["event"])} for c in chips],
                   entry_id=entry_id, notes=notes)


def chips_available(chips_used: list[dict], next_gw: int) -> dict[int, set[str]]:
    """{set number: chips still unused}.  Set 1 covers GW1-19, set 2 GW20+."""
    from model.config import CHIPS
    avail = {1: set(CHIPS), 2: set(CHIPS)}
    for c in chips_used:
        s = 1 if int(c["event"]) <= FIRST_HALF_LAST_GW else 2
        avail[s].discard(c["name"])
    if next_gw > FIRST_HALF_LAST_GW:
        avail[1] = set()                      # first set has expired
    return avail
