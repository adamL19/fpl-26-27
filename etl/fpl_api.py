"""Thin client for the (unofficial, unauthenticated) Fantasy Premier League API.

Only public GET endpoints are used.  Responses can be cached on disk so that
repeated runs don't hammer the server (element summaries are ~600 requests).
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class FPLAPIError(RuntimeError):
    pass


class FPLClient:
    BASE = "https://fantasy.premierleague.com/api/"

    def __init__(self, cache_dir: str | Path | None = None, cache_ttl: float = 0,
                 delay: float = 0.15, timeout: float = 20):
        """cache_ttl is in seconds; 0 disables reading from the cache."""
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = cache_ttl
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        retry = Retry(total=4, backoff_factor=1.0,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET",))
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update(
            {"User-Agent": "fpl-live-dashboard/0.1 (personal analytics project)"})

    # ------------------------------------------------------------------
    def _cache_file(self, path: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.md5(path.encode()).hexdigest() + ".json")

    def get(self, path: str) -> Any:
        cf = self._cache_file(path)
        if cf and self.cache_ttl > 0 and cf.exists():
            if time.time() - cf.stat().st_mtime < self.cache_ttl:
                return json.loads(cf.read_text(encoding="utf-8"))
        try:
            resp = self.session.get(self.BASE + path, timeout=self.timeout)
        except requests.RequestException as exc:
            raise FPLAPIError(f"Request failed for {path}: {exc}") from exc
        if resp.status_code != 200:
            hint = " (the game is probably mid-update; try again in a few minutes)" \
                if resp.status_code == 503 else ""
            raise FPLAPIError(f"HTTP {resp.status_code} for {path}{hint}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise FPLAPIError(f"Non-JSON response for {path}") from exc
        if cf:
            cf.write_text(json.dumps(data), encoding="utf-8")
        time.sleep(self.delay)
        return data

    # ---- league-wide endpoints --------------------------------------
    def bootstrap(self) -> dict:
        return self.get("bootstrap-static/")

    def fixtures(self) -> list:
        return self.get("fixtures/")

    def element_summary(self, element_id: int) -> dict:
        return self.get(f"element-summary/{int(element_id)}/")

    # ---- manager endpoints ------------------------------------------
    def entry(self, entry_id: int) -> dict:
        return self.get(f"entry/{int(entry_id)}/")

    def entry_history(self, entry_id: int) -> dict:
        return self.get(f"entry/{int(entry_id)}/history/")

    def entry_transfers(self, entry_id: int) -> list:
        return self.get(f"entry/{int(entry_id)}/transfers/")

    def entry_picks(self, entry_id: int, event: int) -> dict:
        return self.get(f"entry/{int(entry_id)}/event/{int(event)}/picks/")
