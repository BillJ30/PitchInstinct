"""
PitchInstinct: Spatial Analytics Engine  (TRACKING-DATA ANALYTICS)

Modern football's spatial layer normally comes from optical tracking: a
computer-vision pipeline watches broadcast/tactical-cam footage and emits the
(x, y) position of all 22 players + the ball, frame by frame. StatsBomb ships a
sampled version of exactly that signal in `shot_freeze_frame`: at the instant of
every shot, the located position of every visible player (teammate flag +
position + whether they're the keeper).

This module treats those freeze-frames as the tracking layer and computes the
spatial models a tracking system enables:

  • Pitch control / space-occupation surface  (Voronoi-style dominance grid)
  • Defensive compactness & line height        (shape metrics per shot)
  • Shot-pressure index                        (defenders in the cone, GK angle)
  • Goalkeeper positioning vs the shot bisector (sweeper / angle coverage)
  • Attacking shape: width, depth, centroid

No ML here, these are geometric/spatial computations. The learned xG model in
predictive.py *consumes* features derived from the same freeze-frames, so the
two modules share one positional foundation.

spatial_cv.py adds the literal image side (detecting players & pitch lines from
a frame), closing the loop from pixels → coordinates → these analytics.
"""

import numpy as np
import pandas as pd
import re
from scipy.spatial import Voronoi

PITCH_X, PITCH_Y = 120.0, 80.0
GOAL_X, GOAL_Y = 120.0, 40.0


def _xy(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        try:
            return float(v[0]), float(v[1])
        except Exception:
            return np.nan, np.nan
    nums = re.findall(r"-?\d+\.?\d*", str(v))
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return np.nan, np.nan


def parse_freeze_frame(ff, shooter_xy=None):
    """Normalise a freeze-frame into a tidy DataFrame:
       columns = x, y, teammate (bool), is_gk (bool), position, player."""
    rows = []
    if not isinstance(ff, (list, np.ndarray)):
        return pd.DataFrame(columns=["x", "y", "teammate", "is_gk", "position", "player"])
    for p in ff:
        if not isinstance(p, dict):
            continue
        x, y = _xy(p.get("location"))
        if np.isnan(x):
            continue
        rows.append({
            "x": x, "y": y,
            "teammate": bool(p.get("teammate", False)),
            "is_gk": p.get("position", {}).get("name") == "Goalkeeper",
            "position": p.get("position", {}).get("name", ""),
            "player": p.get("player", {}).get("name", ""),
        })
    df = pd.DataFrame(rows)
    if shooter_xy is not None and len(df):
        df = pd.concat([df, pd.DataFrame([{
            "x": shooter_xy[0], "y": shooter_xy[1], "teammate": True,
            "is_gk": False, "position": "Shooter", "player": "(shooter)"}])],
            ignore_index=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# helper: collect all freeze frames for a team's shots (attacking) or against
# ─────────────────────────────────────────────────────────────────────────────

def collect_frames(events: pd.DataFrame, team: str, attacking=True) -> pd.DataFrame:
    """Return one row per located player across all relevant freeze frames.
    attacking=True → frames from `team`'s own shots (their attacking shape +
    the opponent's defensive shape). attacking=False → frames from shots taken
    AGAINST `team` (their defensive shape under fire)."""
    if attacking:
        shots = events[(events["type"] == "Shot") & (events["team"] == team)]
    else:
        shots = events[(events["type"] == "Shot") & (events["team"] != team)]
    shots = shots[shots["shot_freeze_frame"].notna()]

    frames = []
    for sid, (_, s) in enumerate(shots.iterrows()):
        sx, sy = _xy(s["location"])
        ff = parse_freeze_frame(s["shot_freeze_frame"])
        if len(ff) == 0:
            continue
        ff = ff.copy()
        ff["shot_id"] = sid
        ff["shot_x"] = sx
        ff["shot_y"] = sy
        ff["is_goal"] = 1 if s.get("shot_outcome") == "Goal" else 0
        frames.append(ff)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PITCH-CONTROL / SPACE-OCCUPATION SURFACE
# ═════════════════════════════════════════════════════════════════════════════

def pitch_control_grid(frame_df: pd.DataFrame, nx=48, ny=32, beta=0.30):
    """Soft Voronoi pitch-control surface from one freeze-frame.

    For each grid cell we compute a softmin over players of distance, weighted by
    team (+1 attacking teammate, −1 defender). Output in [-1, 1]: +1 = attacking
    team fully controls that space, −1 = defending team controls it.
    """
    gx = np.linspace(0, PITCH_X, nx)
    gy = np.linspace(0, PITCH_Y, ny)
    XX, YY = np.meshgrid(gx, gy)

    mates = frame_df[frame_df["teammate"]]
    opps = frame_df[~frame_df["teammate"]]
    if len(mates) == 0 or len(opps) == 0:
        return XX, YY, np.zeros_like(XX)

    def influence(players):
        infl = np.zeros_like(XX)
        for _, p in players.iterrows():
            d = np.hypot(XX - p["x"], YY - p["y"])
            infl += np.exp(-beta * d)        # closer players exert more control
        return infl

    a = influence(mates)
    d = influence(opps)
    control = (a - d) / (a + d + 1e-9)
    return XX, YY, control


def control_summary(frame_df: pd.DataFrame) -> dict:
    """Headline pitch-control numbers for one frame."""
    XX, YY, ctrl = pitch_control_grid(frame_df)
    att_share = float((ctrl > 0).mean()) * 100
    # final-third control (attacking third = x>80)
    mask = XX > 80
    final_third = float((ctrl[mask] > 0).mean()) * 100 if mask.any() else 0.0
    return {"attacking_control_pct": round(att_share, 1),
            "final_third_control_pct": round(final_third, 1),
            "grid": (XX, YY, ctrl)}


# ═════════════════════════════════════════════════════════════════════════════
# 2.  DEFENSIVE SHAPE: compactness, line height, width  (per shot, aggregated)
# ═════════════════════════════════════════════════════════════════════════════

def defensive_shape(events: pd.DataFrame, team: str, when_defending=True) -> dict:
    """Average defensive block geometry for `team` measured at the moment shots
    are taken against them (when_defending=True). Uses the OPPONENT-labelled
    players inside each frame (i.e. `team`'s players defending)."""
    frames = collect_frames(events, team, attacking=not when_defending)
    if len(frames) == 0:
        return {}

    # In frames from shots AGAINST team, team's players are the NON-teammates
    # (teammate flag is relative to the shooter, who is the opponent).
    defenders = frames[~frames["teammate"]] if when_defending else frames[frames["teammate"]]
    defenders = defenders[~defenders["is_gk"]]

    per_shot = []
    for sid, g in defenders.groupby("shot_id"):
        if len(g) < 4:
            continue
        # convert to "distance from own goal" so higher = higher line.
        # team defends the x=120 goal in these frames, so own goal is x=120.
        line_height = 120 - g["x"].mean()           # avg yards from own goal
        depth = g["x"].max() - g["x"].min()          # vertical compactness
        width = g["y"].max() - g["y"].min()          # horizontal spread
        # compactness = mean pairwise distance (lower = more compact)
        pts = g[["x", "y"]].values
        if len(pts) > 1:
            dmat = np.hypot(pts[:, None, 0] - pts[None, :, 0],
                            pts[:, None, 1] - pts[None, :, 1])
            compact = dmat[np.triu_indices(len(pts), 1)].mean()
        else:
            compact = np.nan
        per_shot.append({"line_height": line_height, "depth": depth,
                         "width": width, "compactness": compact,
                         "is_goal": g["is_goal"].iloc[0]})
    ps = pd.DataFrame(per_shot)
    if len(ps) == 0:
        return {}
    return {
        "n_shots": int(len(ps)),
        "avg_line_height": round(float(ps["line_height"].mean()), 1),
        "avg_depth": round(float(ps["depth"].mean()), 1),
        "avg_width": round(float(ps["width"].mean()), 1),
        "avg_compactness": round(float(ps["compactness"].mean()), 1),
        "per_shot": ps,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 3.  SHOT-PRESSURE INDEX  &  GK POSITIONING
# ═════════════════════════════════════════════════════════════════════════════

def shot_pressure_table(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """For each of `team`'s shots: how many defenders sat in the shooting cone,
    how close the nearest one was, and where the keeper stood relative to the
    shot→goal bisector. These are the raw spatial pressure signals."""
    shots = events[(events["type"] == "Shot") & (events["team"] == team)]
    shots = shots[shots["shot_freeze_frame"].notna()]
    rows = []
    for _, s in shots.iterrows():
        sx, sy = _xy(s["location"])
        if np.isnan(sx):
            continue
        ff = parse_freeze_frame(s["shot_freeze_frame"])
        opps = ff[~ff["teammate"]]
        defs = opps[~opps["is_gk"]]
        gk = opps[opps["is_gk"]]

        # defenders in the cone (within 6 yd of the shot→goal line, goal-side)
        line = np.array([GOAL_X - sx, GOAL_Y - sy], dtype=float)
        L = np.linalg.norm(line) + 1e-9
        in_cone = 0
        nearest = np.nan
        for _, d in defs.iterrows():
            pv = np.array([d["x"] - sx, d["y"] - sy], dtype=float)
            if d["x"] >= sx:
                perp = abs(np.cross(line, pv)) / L
                if perp < 6:
                    in_cone += 1
            dd = float(np.hypot(d["x"] - sx, d["y"] - sy))
            nearest = dd if np.isnan(nearest) else min(nearest, dd)

        # GK lateral deviation from the shot→goal-centre line at the goal mouth
        gk_dev = np.nan
        if len(gk):
            gx_, gy_ = gk.iloc[0]["x"], gk.iloc[0]["y"]
            # y the keeper "should" be at to sit on the bisector at x=120
            if (GOAL_X - sx) != 0:
                t = (120 - sx) / (GOAL_X - sx)
                ideal_y = sy + t * (GOAL_Y - sy)
            else:
                ideal_y = GOAL_Y
            gk_dev = float(abs(gy_ - ideal_y))

        dist, _ = (float(np.hypot(GOAL_X - sx, sy - GOAL_Y)), None)
        rows.append({
            "player": s.get("player"), "match_id": s.get("match_id"),
            "shot_x": sx, "shot_y": sy, "distance": round(dist, 1),
            "defenders_in_cone": in_cone,
            "nearest_defender": round(nearest, 1) if not np.isnan(nearest) else np.nan,
            "gk_off_line": round(gk_dev, 1) if not np.isnan(gk_dev) else np.nan,
            "xg": s.get("shot_statsbomb_xg", np.nan),
            "is_goal": 1 if s.get("shot_outcome") == "Goal" else 0,
        })
    return pd.DataFrame(rows)


def pressure_overview(pressure_df: pd.DataFrame) -> dict:
    """Summarise the spatial pressure table + split conversion by congestion."""
    if len(pressure_df) == 0:
        return {}
    df = pressure_df
    clear = df[df["defenders_in_cone"] == 0]
    crowded = df[df["defenders_in_cone"] >= 2]
    return {
        "n_shots": int(len(df)),
        "avg_defenders_in_cone": round(float(df["defenders_in_cone"].mean()), 2),
        "avg_nearest_defender": round(float(df["nearest_defender"].mean()), 1),
        "clear_sight_pct": round(float((df["defenders_in_cone"] == 0).mean()) * 100, 1),
        "conv_clear": round(float(clear["is_goal"].mean()) * 100, 1) if len(clear) else 0.0,
        "conv_crowded": round(float(crowded["is_goal"].mean()) * 100, 1) if len(crowded) else 0.0,
        "n_clear": int(len(clear)), "n_crowded": int(len(crowded)),
    }


# ═════════════════════════════════════════════════════════════════════════════
# 4.  PICK A SHOWCASE FRAME (e.g. a goal) for the visual
# ═════════════════════════════════════════════════════════════════════════════

def representative_frame(events: pd.DataFrame, team: str, goals_only=True):
    """Return (shot_row, parsed_frame_df) for a clean showcase, by default the
    team goal with the most located players in its freeze-frame."""
    shots = events[(events["type"] == "Shot") & (events["team"] == team)]
    if goals_only:
        g = shots[shots["shot_outcome"] == "Goal"]
        shots = g if len(g) else shots
    shots = shots[shots["shot_freeze_frame"].notna()]
    best = None; best_n = -1; best_ff = None
    for _, s in shots.iterrows():
        ff = parse_freeze_frame(s["shot_freeze_frame"], shooter_xy=_xy(s["location"]))
        if len(ff) > best_n:
            best_n, best, best_ff = len(ff), s, ff
    return best, best_ff


def voronoi_regions(frame_df: pd.DataFrame):
    """Compute a Voronoi diagram of all players in the frame (for drawing the
    territorial map). Returns the scipy Voronoi object + the point/team arrays."""
    pts = frame_df[["x", "y"]].values
    if len(pts) < 4:
        return None, pts, frame_df["teammate"].values
    try:
        vor = Voronoi(pts)
    except Exception:
        return None, pts, frame_df["teammate"].values
    return vor, pts, frame_df["teammate"].values


# ═════════════════════════════════════════════════════════════════════════════
# 5.  POSSESSION REPLAY: a real, located event-by-event "clip" of a goal
# Built only from genuine StatsBomb fields (location, *_end_location, duration,
# possession, play_pattern). No tracking data exists in this dataset between
# events, so this is honestly an event trail, not a frame-by-frame video, we
# don't invent positions for the seconds in between.
# ═════════════════════════════════════════════════════════════════════════════

PITCH_X_M, PITCH_Y_M = 105.0, 68.0   # real-world pitch used to scale the 120x80 unit grid

def _to_metres(dx, dy):
    """Scale a StatsBomb-unit displacement to an approximate real-world distance (m)."""
    return float(np.hypot(dx * PITCH_X_M / PITCH_X, dy * PITCH_Y_M / PITCH_Y))


def city_goal_moves(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """One row per goal `team` scored this season: match, possession id, minute,
    scorer and play pattern, used to populate the Play Replay picker."""
    shots = events[(events["type"] == "Shot") & (events["team"] == team) &
                    (events["shot_outcome"] == "Goal")]
    rows = []
    for _, s in shots.iterrows():
        rows.append({
            "match_id": s["match_id"], "possession": s["possession"],
            "minute": int(s["minute"]), "second": int(s["second"]),
            "player": s.get("player", "Unknown"), "play_pattern": s.get("play_pattern", ""),
        })
    return pd.DataFrame(rows).sort_values(["match_id", "minute"]).reset_index(drop=True)


def build_possession_replay(events: pd.DataFrame, team: str, match_id, possession):
    """Ordered, located steps of `team`'s play within one possession, ending in
    the shot. Returns (steps, freeze_frame_df_or_None).

    Each step: type, player, minute, second, start (x,y), end (x,y),
    distance_m, duration_s, speed_kmh (None if duration unknown)."""
    ev = events[(events["match_id"] == match_id) & (events["possession"] == possession) &
                (events["team"] == team)].sort_values("index")
    ev = ev[ev["type"].isin(["Pass", "Carry", "Dribble", "Shot"])]

    end_col = {"Pass": "pass_end_location", "Carry": "carry_end_location",
               "Shot": "shot_end_location"}
    steps = []
    freeze_df = None
    for _, r in ev.iterrows():
        sx, sy = _xy(r["location"])
        if np.isnan(sx):
            continue
        col = end_col.get(r["type"])
        ex, ey = _xy(r.get(col)) if col else (sx, sy)
        if np.isnan(ex):
            ex, ey = sx, sy
        dist_m = round(_to_metres(ex - sx, ey - sy), 1)
        dur = r.get("duration")
        dur = float(dur) if pd.notna(dur) and dur and dur > 0 else None
        speed_kmh = round(dist_m / dur * 3.6, 1) if dur else None
        steps.append({
            "type": r["type"], "player": r.get("player", "Unknown"),
            "minute": int(r["minute"]), "second": int(r["second"]),
            "start": (sx, sy), "end": (ex, ey),
            "distance_m": dist_m, "duration_s": dur, "speed_kmh": speed_kmh,
        })
        ff_raw = r.get("shot_freeze_frame")
        if r["type"] == "Shot" and isinstance(ff_raw, (list, np.ndarray)):
            freeze_df = parse_freeze_frame(ff_raw, shooter_xy=(sx, sy))

    return steps, freeze_df
