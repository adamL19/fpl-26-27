"""Team attack/defence ratings used to turn a fixture into expected goals.

Poisson model: goals_for(T vs O, venue) ~ mu_venue * att[T] * def[O], where
`def` > 1 means a leaky defence.  Ratings are fitted on finished fixtures with a
gamma-style prior built from FPL's own team strength numbers, so early in the
season (few games) the prior dominates and later the results take over.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model.config import ModelParams

DEFAULT_MU_HOME = 1.50
DEFAULT_MU_AWAY = 1.20


@dataclass
class TeamModel:
    att: dict[int, float]
    dfn: dict[int, float]
    mu_home: float
    mu_away: float

    @property
    def mu_bar(self) -> float:
        return (self.mu_home + self.mu_away) / 2

    def lam(self, team: int, opp: int, is_home: bool) -> tuple[float, float]:
        """(expected goals for `team`, expected goals against) in this fixture."""
        mu_f = self.mu_home if is_home else self.mu_away
        mu_a = self.mu_away if is_home else self.mu_home
        return (mu_f * self.att[team] * self.dfn[opp],
                mu_a * self.att[opp] * self.dfn[team])


def _prior_from_strength(teams: pd.DataFrame, params: ModelParams):
    t = teams.set_index("id")
    a = (t["strength_attack_home"] + t["strength_attack_away"]) / 2
    d = (t["strength_defence_home"] + t["strength_defence_away"]) / 2
    if a.isna().any() or d.isna().any() or a.mean() == 0 or d.mean() == 0:
        one = pd.Series(1.0, index=t.index)
        return one, one
    att = (a / a.mean()) ** params.team_strength_exponent
    # FPL defence strength: higher = better defence -> leakiness is the inverse
    dfn = (d.mean() / d) ** params.team_strength_exponent
    return att / att.mean(), dfn / dfn.mean()


def fit_team_model(teams: pd.DataFrame, fixtures: pd.DataFrame,
                   params: ModelParams | None = None, iterations: int = 30) -> TeamModel:
    params = params or ModelParams()
    ids = list(teams["id"])
    prior_att, prior_def = _prior_from_strength(teams, params)

    done = fixtures[(fixtures["finished"] == 1)
                    & fixtures["team_h_score"].notna() & fixtures["team_a_score"].notna()]
    n = len(done)
    if n:
        obs_h = done["team_h_score"].mean()
        obs_a = done["team_a_score"].mean()
        w = n / (n + 30.0)
        mu_h = w * obs_h + (1 - w) * DEFAULT_MU_HOME
        mu_a = w * obs_a + (1 - w) * DEFAULT_MU_AWAY
    else:
        mu_h, mu_a = DEFAULT_MU_HOME, DEFAULT_MU_AWAY
    mu_bar = (mu_h + mu_a) / 2
    k = params.team_prior_games

    att = {t: float(prior_att[t]) for t in ids}
    dfn = {t: float(prior_def[t]) for t in ids}
    if n:
        h = done["team_h"].to_numpy(); a_ = done["team_a"].to_numpy()
        hs = done["team_h_score"].to_numpy(float); as_ = done["team_a_score"].to_numpy(float)
        for _ in range(iterations):
            new_att, new_def = {}, {}
            for t in ids:
                home = h == t; away = a_ == t
                gf = hs[home].sum() + as_[away].sum()
                exp_f = sum(mu_h * dfn[o] for o in a_[home]) + sum(mu_a * dfn[o] for o in h[away])
                ga = as_[home].sum() + hs[away].sum()
                exp_a = sum(mu_a * att[o] for o in a_[home]) + sum(mu_h * att[o] for o in h[away])
                new_att[t] = (gf + k * prior_att[t] * mu_bar) / (exp_f + k * mu_bar)
                new_def[t] = (ga + k * prior_def[t] * mu_bar) / (exp_a + k * mu_bar)
            ma = np.mean(list(new_att.values())); md = np.mean(list(new_def.values()))
            att = {t: v / ma for t, v in new_att.items()}
            dfn = {t: v / md for t, v in new_def.items()}
    return TeamModel(att=att, dfn=dfn, mu_home=mu_h, mu_away=mu_a)
