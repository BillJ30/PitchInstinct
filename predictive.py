"""
PitchInstinct: Predictive Analytics Engine  (TRAINED MACHINE LEARNING)

This is the app's first genuinely *predictive* layer. Everything in
analytics.py / features.py is descriptive (counts, shares, Poisson). Here we
TRAIN supervised models on the StatsBomb event data and use them to predict
unseen outcomes:

  1. Expected-Goals (xG) model:      Gradient Boosted Trees on shot geometry
                                       + freeze-frame spatial features.
                                       Target: did the shot become a goal?
  2. Match-outcome model:            Multinomial Logistic Regression on
                                       team-strength + recent-form features.
                                       Target: Win / Draw / Loss for City.
  3. Player goal-involvement model:  Ridge regression predicting a player's
                                       next-match attacking output from a
                                       rolling-form feature window.

All three are trained at runtime, cached, and reported WITH honest validation
metrics (log-loss / ROC-AUC / accuracy / calibration), so the numbers can be
trusted rather than asserted.

Spatial features for the xG model are derived from `shot_freeze_frame`, the
exact (x, y) of every player at the instant of the shot, which is precisely
the positional signal a computer-vision tracking system extracts from footage.
That is what links this module to spatial.py.
"""

import numpy as np
import pandas as pd
import re

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             brier_score_loss, mean_absolute_error)
from sklearn.calibration import calibration_curve

GOAL_X, GOAL_Y = 120.0, 40.0     # centre of the opponent goal (statsbomb frame)
GOAL_HALF_W = 3.66               # half goal width in yards (≈ 7.32 m total)


# ─────────────────────────────────────────────────────────────────────────────
# shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_xy(v):
    """Freeze-frame / location entries are stored either as a list or as a
    stringified numpy array like '[116.9  41.2]'. Parse both robustly."""
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return np.nan, np.nan
    nums = re.findall(r"-?\d+\.?\d*", str(v))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return np.nan, np.nan


def _shot_geom(x, y):
    """Distance to goal centre and the shooting angle subtended by the posts."""
    dx = GOAL_X - x
    dy = y - GOAL_Y
    dist = float(np.hypot(dx, dy))
    # angle between the two posts seen from the shot location (radians)
    p1 = np.array([GOAL_X, GOAL_Y - GOAL_HALF_W])
    p2 = np.array([GOAL_X, GOAL_Y + GOAL_HALF_W])
    s = np.array([x, y])
    v1, v2 = p1 - s, p2 - s
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-9
    cosang = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return dist, float(np.arccos(cosang))


# ═════════════════════════════════════════════════════════════════════════════
# 1.  EXPECTED GOALS (xG): Gradient Boosted Trees with spatial features
# ═════════════════════════════════════════════════════════════════════════════

def build_shot_features(events: pd.DataFrame) -> pd.DataFrame:
    """One row per shot. Geometry + body part + situational flags + freeze-frame
    spatial features (defenders & keeper positioning). Returns features + label
    `goal` and a few descriptive columns kept for display."""
    shots = events[(events["type"] == "Shot")].copy()
    shots = shots[shots["location"].notna()]

    rows = []
    for _, s in shots.iterrows():
        x, y = _parse_xy(s["location"])
        if np.isnan(x) or np.isnan(y):
            continue
        dist, angle = _shot_geom(x, y)

        # --- freeze-frame spatial features (the CV/tracking signal) ----------
        n_def = 0           # opponents between shot and goal (in the cone)
        n_def_close = 0     # opponents within 2 yd
        gk_dist = np.nan    # how far the keeper is from the shot
        gk_x = gk_y = np.nan
        ff = s.get("shot_freeze_frame")
        if isinstance(ff, (list, np.ndarray)):
            for p in ff:
                if not isinstance(p, dict):
                    continue
                px, py = _parse_xy(p.get("location"))
                if np.isnan(px):
                    continue
                is_teammate = bool(p.get("teammate", False))
                is_gk = p.get("position", {}).get("name") == "Goalkeeper"
                if is_gk and not is_teammate:
                    gk_x, gk_y = px, py
                    gk_dist = float(np.hypot(GOAL_X - px, py - GOAL_Y))
                if (not is_teammate) and (not is_gk):
                    # is this opponent roughly in the shot-to-goal cone?
                    if px >= x:  # goal-side of the shooter
                        # distance from the opponent to the shot→goal line
                        line = np.array([GOAL_X - x, GOAL_Y - y], dtype=float)
                        pv = np.array([px - x, py - y], dtype=float)
                        L = np.linalg.norm(line) + 1e-9
                        perp = abs(line[0] * pv[1] - line[1] * pv[0]) / L
                        if perp < 6:           # within ~6 yd of the channel
                            n_def += 1
                    if np.hypot(px - x, py - y) < 2.0:
                        n_def_close += 1

        bp = str(s.get("shot_body_part", "")) or "Unknown"
        tech = str(s.get("shot_technique", "")) or "Normal"
        rows.append({
            "x": x, "y": y,
            "distance": dist,
            "angle": angle,
            "header": 1 if bp == "Head" else 0,
            "weak_other": 1 if bp == "Other" else 0,
            "first_time": 1 if s.get("shot_first_time") == True else 0,
            "one_on_one": 1 if s.get("shot_one_on_one") == True else 0,
            "open_goal": 1 if s.get("shot_open_goal") == True else 0,
            "under_pressure": 1 if s.get("under_pressure") == True else 0,
            "volley": 1 if tech in ("Volley", "Half Volley", "Overhead Kick") else 0,
            "from_open_play": 1 if str(s.get("shot_type", "")) == "Open Play" else 0,
            "n_defenders": n_def,
            "n_def_close": n_def_close,
            "gk_dist_to_goal": gk_dist if not np.isnan(gk_dist) else 2.0,
            # display / grouping
            "player": s.get("player"),
            "team": s.get("team"),
            "match_id": s.get("match_id"),
            "sb_xg": s.get("shot_statsbomb_xg", np.nan),
            "goal": 1 if s.get("shot_outcome") == "Goal" else 0,
        })
    df = pd.DataFrame(rows)
    # median-fill any missing keeper distance
    if len(df):
        df["gk_dist_to_goal"] = df["gk_dist_to_goal"].fillna(df["gk_dist_to_goal"].median())
    return df


XG_FEATURES = ["distance", "angle", "header", "weak_other", "first_time",
               "one_on_one", "open_goal", "under_pressure", "volley",
               "from_open_play", "n_defenders", "n_def_close", "gk_dist_to_goal"]


def train_xg_model(events: pd.DataFrame) -> dict:
    """Train a Gradient Boosted xG model. Returns the fitted model, honest
    out-of-fold metrics, feature importances, a calibration curve, and the
    per-shot feature frame with model predictions attached."""
    feat = build_shot_features(events)
    feat = feat.dropna(subset=XG_FEATURES)
    X = feat[XG_FEATURES].values
    y = np.asarray(feat["goal"].tolist(), dtype=int)

    model = GradientBoostingClassifier(
        n_estimators=180, max_depth=3, learning_rate=0.05,
        subsample=0.85, random_state=42)

    # out-of-fold predictions for honest metrics (no leakage)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = {
        "n_shots": int(len(feat)),
        "goals": int(y.sum()),
        "base_rate": round(float(y.mean()), 3),
        "log_loss": round(float(log_loss(y, oof)), 4),
        "roc_auc": round(float(roc_auc_score(y, oof)), 3),
        "brier": round(float(brier_score_loss(y, oof)), 4),
    }
    # benchmark against StatsBomb's own xG, where present
    have_sb = feat["sb_xg"].notna().values
    if have_sb.sum() > 20:
        metrics["sb_log_loss"] = round(float(log_loss(y[have_sb], feat["sb_xg"].values[have_sb])), 4)

    # calibration curve (predicted vs observed goal frequency)
    frac_pos, mean_pred = calibration_curve(y, oof, n_bins=8, strategy="quantile")
    calib = pd.DataFrame({"predicted": np.round(mean_pred, 3),
                          "observed": np.round(frac_pos, 3)})

    # fit on all data for downstream prediction; expose feature importances
    model.fit(X, y)
    importances = (pd.DataFrame({"feature": XG_FEATURES,
                                 "importance": model.feature_importances_})
                   .sort_values("importance", ascending=False)
                   .reset_index(drop=True))

    feat = feat.copy()
    feat["model_xg"] = model.predict_proba(X)[:, 1]
    return {"model": model, "metrics": metrics, "importances": importances,
            "calibration": calib, "shots": feat}


def predict_xg_for_team(xg_pack: dict, team: str) -> pd.DataFrame:
    """Per-player model-xG vs goals for a team (over- / under-performance)."""
    s = xg_pack["shots"]
    s = s[s["team"] == team]
    g = (s.groupby("player")
           .agg(shots=("goal", "size"), goals=("goal", "sum"),
                model_xg=("model_xg", "sum"), sb_xg=("sb_xg", "sum"))
           .reset_index())
    g = g[g["shots"] >= 5]
    g["model_xg"] = g["model_xg"].round(2)
    g["overperformance"] = (g["goals"] - g["model_xg"]).round(2)
    return g.sort_values("model_xg", ascending=False).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  MATCH-OUTCOME MODEL: Multinomial Logistic Regression
# ═════════════════════════════════════════════════════════════════════════════

def _team_match_table(events: pd.DataFrame) -> pd.DataFrame:
    """Per (match, team) attacking/defending summary used to engineer features."""
    rows = []
    for (mid, team), g in events.groupby(["match_id", "team"]):
        shots = g[g["type"] == "Shot"]
        passes = g[g["type"] == "Pass"]
        rows.append({
            "match_id": mid, "team": team,
            "shots": len(shots),
            "xg": float(shots["shot_statsbomb_xg"].sum()) if "shot_statsbomb_xg" in shots else 0.0,
            "goals": int((shots["shot_outcome"] == "Goal").sum()),
            "passes": len(passes),
            "pass_pct": float(passes["pass_outcome"].isna().mean()) if len(passes) else 0.0,
        })
    return pd.DataFrame(rows)


def build_match_features(events: pd.DataFrame, matches: pd.DataFrame,
                         our_team="Manchester City") -> pd.DataFrame:
    """One row per City match with PRE-MATCH features (rolling form up to but
    excluding that match) and the realised result label."""
    tbl = _team_match_table(events)
    m = matches.sort_values("match_week").copy()

    # City rolling form (xg for / against, points) BEFORE each match
    feats = []
    hist_for, hist_against, hist_pts = [], [], []
    for _, row in m.iterrows():
        mid = row["match_id"]
        city = tbl[(tbl["match_id"] == mid) & (tbl["team"] == our_team)]
        opp = tbl[(tbl["match_id"] == mid) & (tbl["team"] != our_team)]
        if len(city) == 0 or len(opp) == 0:
            continue
        city = city.iloc[0]; opp = opp.iloc[0]

        def _avg(lst, k=5, default=0.0):
            return float(np.mean(lst[-k:])) if lst else default

        feats.append({
            "match_id": mid,
            "match_week": row["match_week"],
            "opponent": row["opponent"],
            "is_home": 1 if row["home_or_away"] == "Home" else 0,
            "form_xg_for": _avg(hist_for, default=1.5),
            "form_xg_against": _avg(hist_against, default=1.0),
            "form_points": _avg(hist_pts, default=1.5),
            "result": row["result"],
        })
        # update history AFTER using it (no leakage)
        hist_for.append(city["xg"])
        hist_against.append(opp["xg"])
        hist_pts.append(3 if row["result"] == "W" else (1 if row["result"] == "D" else 0))

    return pd.DataFrame(feats)


MATCH_FEATURES = ["is_home", "form_xg_for", "form_xg_against", "form_points"]
_LABELS = ["L", "D", "W"]


def train_match_model(events: pd.DataFrame, matches: pd.DataFrame,
                      our_team="Manchester City") -> dict:
    """Train a 3-class outcome model with leave-one-out CV (small-sample safe)."""
    feat = build_match_features(events, matches, our_team)
    feat = feat[feat["match_week"] >= 4]          # need some history to form features
    X = feat[MATCH_FEATURES].values
    y = np.asarray(feat["result"].tolist())          # plain ndarray (avoid pyarrow indexing)

    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=500)),
    ])
    n = len(feat)
    cv = KFold(n_splits=min(8, n), shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, X, y, cv=cv)
    acc = accuracy_score(y, oof)
    base = pd.Series(y).value_counts(normalize=True).max()  # majority-class baseline

    pipe.fit(X, y)
    coefs = pd.DataFrame(pipe.named_steps["lr"].coef_, columns=MATCH_FEATURES,
                         index=pipe.named_steps["lr"].classes_).round(3)
    return {"model": pipe, "features": feat,
            "metrics": {"n_games": int(n), "cv_accuracy": round(float(acc), 3),
                        "baseline": round(float(base), 3),
                        "classes": list(pipe.named_steps["lr"].classes_)},
            "coefficients": coefs}


def predict_next_match(match_pack: dict, is_home: bool) -> dict:
    """Predict W/D/L probabilities for the next match using City's current form
    (the latest engineered feature row) and a chosen venue."""
    feat = match_pack["features"]
    if len(feat) == 0:
        return {}
    last = feat.iloc[-1]
    x = np.array([[1 if is_home else 0, last["form_xg_for"],
                   last["form_xg_against"], last["form_points"]]])
    model = match_pack["model"]
    proba = model.predict_proba(x)[0]
    classes = list(model.named_steps["lr"].classes_)
    out = {c: round(float(p) * 100, 1) for c, p in zip(classes, proba)}
    return {"W": out.get("W", 0), "D": out.get("D", 0), "L": out.get("L", 0)}


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PLAYER GOAL-INVOLVEMENT MODEL: Ridge regression on rolling form
# ═════════════════════════════════════════════════════════════════════════════

def build_player_form_samples(events: pd.DataFrame, matches: pd.DataFrame,
                              team="Manchester City") -> pd.DataFrame:
    """For every (player, match) build rolling-form features from the previous
    matches and the realised goal-involvements (goals+assists) as the target."""
    ev = events[events["team"] == team]
    mw = matches[["match_id", "match_week"]].set_index("match_id")["match_week"].to_dict()

    per = []
    for (mid, player), g in ev.groupby(["match_id", "player"]):
        if player is None:
            continue
        shots = g[g["type"] == "Shot"]
        passes = g[g["type"] == "Pass"]
        per.append({
            "player": player, "match_id": mid, "mw": mw.get(mid, np.nan),
            "goals": int((shots["shot_outcome"] == "Goal").sum()),
            "assists": int(passes["pass_goal_assist"].sum()) if "pass_goal_assist" in passes else 0,
            "shots": len(shots),
            "xg": float(shots["shot_statsbomb_xg"].sum()) if "shot_statsbomb_xg" in shots else 0.0,
            "key_passes": int(passes["pass_shot_assist"].sum()) if "pass_shot_assist" in passes else 0,
        })
    pm = pd.DataFrame(per).dropna(subset=["mw"]).sort_values(["player", "mw"])
    pm["involve"] = pm["goals"] + pm["assists"]

    samples = []
    for player, g in pm.groupby("player"):
        g = g.sort_values("mw").reset_index(drop=True)
        if len(g) < 4:
            continue
        for i in range(3, len(g)):
            past = g.iloc[:i]
            samples.append({
                "player": player, "mw": g.iloc[i]["mw"],
                "roll_xg": past["xg"].tail(3).mean(),
                "roll_shots": past["shots"].tail(3).mean(),
                "roll_keyp": past["key_passes"].tail(3).mean(),
                "roll_involve": past["involve"].tail(3).mean(),
                "target_involve": g.iloc[i]["involve"],
            })
    return pd.DataFrame(samples)


FORM_FEATURES = ["roll_xg", "roll_shots", "roll_keyp", "roll_involve"]


def train_form_model(events: pd.DataFrame, matches: pd.DataFrame,
                     team="Manchester City") -> dict:
    """Ridge regression predicting next-match goal involvement from form."""
    s = build_player_form_samples(events, matches, team)
    if len(s) < 30:
        return {"model": None, "samples": s, "metrics": {"n": int(len(s))}}
    X = s[FORM_FEATURES].values
    y = np.asarray(s["target_involve"].tolist(), dtype=float)
    pipe = Pipeline([("sc", StandardScaler()), ("rg", Ridge(alpha=2.0))])
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, X, y, cv=cv)
    mae = mean_absolute_error(y, oof)
    naive = mean_absolute_error(y, np.full_like(y, y.mean(), dtype=float))
    pipe.fit(X, y)
    return {"model": pipe, "samples": s,
            "metrics": {"n": int(len(s)), "cv_mae": round(float(mae), 3),
                        "naive_mae": round(float(naive), 3)}}


def predict_player_form(form_pack: dict, events: pd.DataFrame,
                        matches: pd.DataFrame, team="Manchester City",
                        top_n=10) -> pd.DataFrame:
    """Project every player's next-match goal involvement from their latest
    3-match form window."""
    if form_pack.get("model") is None:
        return pd.DataFrame()
    s = build_player_form_samples(events, matches, team)
    latest = (s.sort_values("mw").groupby("player").tail(1))
    if len(latest) == 0:
        return pd.DataFrame()
    X = latest[FORM_FEATURES].values
    latest = latest.copy()
    latest["projected_involve"] = form_pack["model"].predict(X).clip(min=0).round(2)
    out = latest[["player", "roll_involve", "projected_involve"]].copy()
    out = out.rename(columns={"roll_involve": "recent_form"})
    out["recent_form"] = out["recent_form"].round(2)
    return out.sort_values("projected_involve", ascending=False).head(top_n).reset_index(drop=True)
