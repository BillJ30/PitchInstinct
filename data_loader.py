"""
PitchInstinct: Data Loader
Pulls StatsBomb Premier League 2015/16 data and caches locally.
Manchester City as the home team; all 19 opponents available.
"""

import pandas as pd
import numpy as np
import os, json, threading, warnings
warnings.filterwarnings("ignore")

from statsbombpy import sb

COMPETITION_ID = 2   # Premier League
SEASON_ID      = 27  # 2015/16
OUR_TEAM       = "Manchester City"
CACHE_DIR      = os.path.join(os.path.dirname(__file__), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Per-team locks so a background prefetch and a foreground load can't fetch/write the
# same team's parquet at the same moment (see get_team_full_matches).
_team_locks = {}
_team_locks_guard = threading.Lock()

def _team_lock(team: str) -> threading.Lock:
    with _team_locks_guard:
        return _team_locks.setdefault(team, threading.Lock())


# ── helpers ──────────────────────────────────────────────────────────────────

def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def ts_to_sec(ts: str) -> float:
    """Convert HH:MM:SS.mmm to total seconds."""
    try:
        h, m, s = str(ts).split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return np.nan


def loc_x(loc):
    return loc[0] if isinstance(loc, list) and len(loc) >= 2 else np.nan

def loc_y(loc):
    return loc[1] if isinstance(loc, list) and len(loc) >= 2 else np.nan


# ── match list ────────────────────────────────────────────────────────────────

def get_city_matches() -> pd.DataFrame:
    path = _cache_path("matches")
    if os.path.exists(path):
        return pd.read_parquet(path)
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    city = matches[
        (matches["home_team"] == OUR_TEAM) | (matches["away_team"] == OUR_TEAM)
    ].copy()
    city["opponent"] = city.apply(
        lambda r: r["away_team"] if r["home_team"] == OUR_TEAM else r["home_team"], axis=1
    )
    city["home_or_away"] = city.apply(
        lambda r: "Home" if r["home_team"] == OUR_TEAM else "Away", axis=1
    )
    city["city_score"] = city.apply(
        lambda r: r["home_score"] if r["home_team"] == OUR_TEAM else r["away_score"], axis=1
    )
    city["opp_score"] = city.apply(
        lambda r: r["away_score"] if r["home_team"] == OUR_TEAM else r["home_score"], axis=1
    )
    city["result"] = city.apply(
        lambda r: "W" if r["city_score"] > r["opp_score"]
                  else ("D" if r["city_score"] == r["opp_score"] else "L"), axis=1
    )
    city.to_parquet(path)
    return city


# ── raw events (all City matches) ────────────────────────────────────────────

def get_all_events() -> pd.DataFrame:
    path = _cache_path("all_events")
    if os.path.exists(path):
        return pd.read_parquet(path)
    matches = get_city_matches()
    frames = []
    for mid in matches["match_id"]:
        try:
            ev = sb.events(match_id=mid)
            ev["match_id"] = mid
            frames.append(ev)
        except Exception:
            pass
    events = pd.concat(frames, ignore_index=True)
    # Flatten location
    events["loc_x"] = events["location"].apply(loc_x)
    events["loc_y"] = events["location"].apply(loc_y)
    events["ts_sec"] = events["timestamp"].apply(ts_to_sec)
    events.to_parquet(path)
    return events


# ── per-opponent events ───────────────────────────────────────────────────────

def get_opponent_events(opponent: str) -> pd.DataFrame:
    safe = opponent.replace(" ", "_").replace("/","_")
    path = _cache_path(f"opp_{safe}")
    if os.path.exists(path):
        return pd.read_parquet(path)
    matches = get_city_matches()
    opp_matches = matches[matches["opponent"] == opponent]
    frames = []
    for mid in opp_matches["match_id"]:
        try:
            ev = sb.events(match_id=mid)
            ev["match_id"] = mid
            frames.append(ev)
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    events = pd.concat(frames, ignore_index=True)
    events["loc_x"] = events["location"].apply(loc_x)
    events["loc_y"] = events["location"].apply(loc_y)
    events["ts_sec"] = events["timestamp"].apply(ts_to_sec)
    events.to_parquet(path)
    return events


# ── squad helpers ─────────────────────────────────────────────────────────────

def get_city_squad_stats() -> pd.DataFrame:
    """Aggregate season stats for every Man City player."""
    path = _cache_path("squad_stats")
    if os.path.exists(path):
        return pd.read_parquet(path)
    events = get_all_events()
    city_ev = events[events["team"] == OUR_TEAM].copy()

    passes   = city_ev[city_ev["type"] == "Pass"]
    shots    = city_ev[city_ev["type"] == "Shot"]
    carries  = city_ev[city_ev["type"] == "Carry"]
    pressures = city_ev[city_ev["type"] == "Pressure"]

    players = sorted(city_ev["player"].dropna().unique())
    rows = []
    for p in players:
        p_pass   = passes[passes["player"] == p]
        p_shot   = shots[shots["player"] == p]
        p_pres   = pressures[pressures["player"] == p]

        # passes
        total_pass = len(p_pass)
        comp_pass  = p_pass["pass_outcome"].isna().sum()  # NaN = success
        pass_pct   = round(comp_pass / total_pass * 100, 1) if total_pass else 0

        # pressured passes
        pp = p_pass[p_pass["under_pressure"] == True]
        pp_total = len(pp)
        pp_comp  = pp["pass_outcome"].isna().sum()
        pp_pct   = round(pp_comp / pp_total * 100, 1) if pp_total else None

        # shots & goals
        total_shot = len(p_shot)
        goals      = (p_shot["shot_outcome"] == "Goal").sum()
        xg_total   = p_shot["shot_statsbomb_xg"].sum() if "shot_statsbomb_xg" in p_shot.columns else 0

        # assists (pass that leads to shot)
        assists = p_pass["pass_goal_assist"].sum() if "pass_goal_assist" in p_pass.columns else 0

        rows.append({
            "player": p,
            "matches": city_ev[city_ev["player"] == p]["match_id"].nunique(),
            "passes": total_pass,
            "pass_completion_pct": pass_pct,
            "pressured_passes": pp_total,
            "pressured_pass_pct": pp_pct,
            "shots": total_shot,
            "goals": int(goals),
            "xg": round(float(xg_total), 2),
            "assists": int(assists),
            "pressures_applied": len(p_pres),
        })

    df = pd.DataFrame(rows).sort_values("matches", ascending=False)
    df.to_parquet(path)
    return df


def get_opponents() -> list:
    return sorted(get_city_matches()["opponent"].unique())


def get_player_positions():
    """Extract each Man City player's most common position from Starting XI events."""
    path = _cache_path("positions")
    if os.path.exists(path):
        return pd.read_parquet(path)
    events = get_all_events()
    sxi = events[(events["type"] == "Starting XI") & (events["team"] == OUR_TEAM)]
    rows = []
    for _, ev in sxi.iterrows():
        tac = ev["tactics"]
        if isinstance(tac, dict):
            for p in tac.get("lineup", []):
                rows.append({
                    "player": p.get("player", {}).get("name"),
                    "position": p.get("position", {}).get("name"),
                })
    df = pd.DataFrame(rows).dropna()
    # most common position per player
    pos = (df.groupby("player")["position"]
             .agg(lambda s: s.value_counts().index[0])
             .reset_index())
    # simple position group
    def group(p):
        if "Goalkeeper" in p: return "GK"
        if "Back" in p: return "DEF"
        if "Midfield" in p or "Wing" in p: return "MID"
        if "Forward" in p or "Striker" in p or "Center Forward" in p: return "FWD"
        return "MID"
    pos["pos_group"] = pos["position"].apply(group)
    pos.to_parquet(path)
    return pos


# ─────────────────────────────────────────────────────────────────────────────
# FULL-SEASON OPPONENT DATA (for genuine scouting)
# Pulls every match a team played, keeping BOTH teams' events, so we can see
# how the opponent attacks AND where they concede.
# ─────────────────────────────────────────────────────────────────────────────

def get_all_pl_matches():
    path = _cache_path("pl_matches")
    if os.path.exists(path):
        return pd.read_parquet(path)
    m = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    keep = ["match_id", "match_date", "home_team", "away_team",
            "home_score", "away_score"]
    m = m[[c for c in keep if c in m.columns]].copy()
    m.to_parquet(path)
    return m


def get_team_full_matches(team):
    """All events from every match `team` played (both teams' events).

    Thread-safe: guarded by a per-team lock and published with an atomic rename, so
    the background prefetch thread and a foreground load never half-write or double-fetch
    the same parquet."""
    safe = team.replace(" ", "_").replace("/", "_")
    path = _cache_path(f"team_{safe}")
    if os.path.exists(path):
        return pd.read_parquet(path)
    with _team_lock(team):
        if os.path.exists(path):            # another thread may have built it while we waited
            return pd.read_parquet(path)
        allm = get_all_pl_matches()
        tm = allm[(allm["home_team"] == team) | (allm["away_team"] == team)]
        frames = []
        for mid in tm["match_id"]:
            try:
                ev = sb.events(match_id=mid)
                ev["match_id"] = mid
                frames.append(ev)
            except Exception:
                pass
        e = pd.concat(frames, ignore_index=True)
        e["loc_x"] = e["location"].apply(loc_x)
        e["loc_y"] = e["location"].apply(loc_y)
        e["ts_sec"] = e["timestamp"].apply(ts_to_sec)
        e["n_matches"] = tm["match_id"].nunique()
        tmp = path + ".tmp"
        e.to_parquet(tmp)
        os.replace(tmp, path)               # atomic publish: readers see a complete file or none
        return e


def team_full_cached(team) -> bool:
    """True if this team's full-season events are already on disk."""
    safe = team.replace(" ", "_").replace("/", "_")
    return os.path.exists(_cache_path(f"team_{safe}"))


def prefetch_team_full_matches(team) -> bool:
    """Ensure a team's full-season events are cached to disk, fetching from StatsBomb
    only if missing. Used by the app's background prefetch thread; does no Streamlit work
    and skips teams already cached, so it's cheap to call for the whole league."""
    if team_full_cached(team):
        return True
    try:
        get_team_full_matches(team)         # builds + writes the parquet (frame discarded by caller)
        return True
    except Exception:
        return False
