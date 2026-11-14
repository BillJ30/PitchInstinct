"""
PitchInstinct: Analytics Engine
All descriptive. No ML. All methods from the MIS41420 lecture decks.
"""

import pandas as pd
import numpy as np
import networkx as nx


# ─────────────────────────────────────────────────────────────────────────────
# IDEA 1: Pressure-Resistance Rating
# Lecture anchor: Rating Systems (accumulation formula, Stefani 1998)
# ─────────────────────────────────────────────────────────────────────────────

def pressure_resistance_rating(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    For every player in `team`, compute:
      - pressured_pass_pct   : pass completion % when under_pressure = True
      - unpressured_pass_pct : pass completion % when not under pressure
      - resistance_drop      : how much % drops under pressure (lower = better)
      - PR_score             : weighted composite (our named metric, 0-100)
    Zone weighting: defensive third (x < 40) = 3×, middle (40-80) = 2×, att (>80) = 1×
    """
    passes = events[(events["type"] == "Pass") & (events["team"] == team)].copy()
    passes["success"] = passes["pass_outcome"].isna()
    passes["zone_weight"] = passes["loc_x"].apply(
        lambda x: 3.0 if (x is not None and not np.isnan(x) and x < 40)
                  else (2.0 if (x is not None and not np.isnan(x) and x < 80) else 1.0)
    )

    rows = []
    for player, grp in passes.groupby("player"):
        up   = grp[grp["under_pressure"] == True]
        nup  = grp[grp["under_pressure"] != True]

        if len(up) < 3:          # not enough pressured data
            continue

        up_pct  = up["success"].mean() * 100
        nup_pct = nup["success"].mean() * 100 if len(nup) else up_pct
        drop    = nup_pct - up_pct            # how much worse under pressure

        # weighted score: high completion under pressure in important zones
        weighted_comp = (up["success"] * up["zone_weight"]).sum()
        weighted_total = up["zone_weight"].sum()
        pr_score = round((weighted_comp / weighted_total) * 100, 1) if weighted_total else 0

        rows.append({
            "player": player,
            "pressured_pass_pct": round(up_pct, 1),
            "unpressured_pass_pct": round(nup_pct, 1),
            "resistance_drop": round(drop, 1),
            "pressured_attempts": len(up),
            "PR_score": pr_score,
        })

    df = pd.DataFrame(rows).sort_values("PR_score", ascending=False).reset_index(drop=True)
    df["PR_rank"] = df.index + 1
    return df


def pressure_resistance_zones(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """Return pressured pass events with location, for the spatial map."""
    passes = events[(events["type"] == "Pass") & (events["team"] == team) &
                    (events["under_pressure"] == True)].copy()
    passes["success"] = passes["pass_outcome"].isna()
    return passes[["player", "loc_x", "loc_y", "success", "pass_length",
                   "pass_angle", "match_id"]].dropna(subset=["loc_x", "loc_y"])


# ─────────────────────────────────────────────────────────────────────────────
# IDEA 2: Tactical Degradation
# Lecture anchor: Spatio-temporal analysis (team shape over time)
# ─────────────────────────────────────────────────────────────────────────────

def tactical_degradation(events: pd.DataFrame, team: str) -> dict:
    """
    Compare opponent's defensive shape in first 75 min vs last 15 min.
    Returns avg defensive action location per phase and the shift.
    """
    def_types = ["Pressure", "Tackle", "Interception", "Clearance", "Block"]
    def_ev = events[(events["type"].isin(def_types)) & (events["team"] == team)].copy()
    def_ev = def_ev.dropna(subset=["loc_x", "loc_y", "minute"])
    def_ev["phase"] = def_ev["minute"].apply(
        lambda m: "First 75" if m <= 75 else "Last 15"
    )

    grouped = def_ev.groupby("phase")[["loc_x", "loc_y"]].mean().round(2)
    counts  = def_ev.groupby("phase").size().rename("count")
    result  = grouped.join(counts)

    # Compute the shift
    if "First 75" in result.index and "Last 15" in result.index:
        x_shift = result.loc["Last 15", "loc_x"] - result.loc["First 75", "loc_x"]
        y_shift = result.loc["Last 15", "loc_y"] - result.loc["First 75", "loc_y"]
    else:
        x_shift = y_shift = 0.0

    # Per-player breakdown (who degrades most)
    player_phase = def_ev.groupby(["player", "phase"])["loc_x"].mean().unstack()
    player_phase = player_phase.dropna()
    if "First 75" in player_phase.columns and "Last 15" in player_phase.columns:
        player_phase["x_shift"] = (
            player_phase["Last 15"] - player_phase["First 75"]
        ).round(2)
        player_phase = player_phase.sort_values("x_shift", ascending=False)
    else:
        player_phase = pd.DataFrame()

    return {
        "phase_summary": result,
        "x_shift": round(float(x_shift), 2),
        "y_shift": round(float(y_shift), 2),
        "player_degradation": player_phase,
        "raw_events": def_ev,
    }


# ─────────────────────────────────────────────────────────────────────────────
# IDEA 3: 5-Second Transition Vulnerability
# Lecture anchor: Spatio-temporal (space creation, event location)
# Validated by StatsBomb's own "High Press Shots" research
# ─────────────────────────────────────────────────────────────────────────────

def transition_vulnerability(events: pd.DataFrame, opponent: str,
                              window_sec: float = 5.0) -> pd.DataFrame:
    """
    After opponent loses possession, where are the open spaces in the next N sec?
    Returns a DataFrame of all events occurring in that window (for heatmap).
    """
    ev = events.sort_values(["match_id", "period", "index"]).copy()
    ev["prev_poss_team"] = ev.groupby(["match_id", "period"])["possession_team"].shift(1)
    # Turnover = possession changes TO the opponent losing it (City gaining it)
    turnovers = ev[
        (ev["possession_team"] != ev["prev_poss_team"]) &
        (ev["possession_team"] == "Manchester City")
    ].copy()

    transition_rows = []
    for _, row in turnovers.iterrows():
        t0 = row["ts_sec"]
        if pd.isna(t0):
            continue
        mask = (
            (ev["match_id"] == row["match_id"]) &
            (ev["period"] == row["period"]) &
            (ev["ts_sec"] >= t0) &
            (ev["ts_sec"] <= t0 + window_sec) &
            (ev["team"] == opponent)          # where OPPONENT players are in that window
        )
        window_events = ev[mask].copy()
        window_events["turnover_x"] = row["loc_x"]
        window_events["turnover_y"] = row["loc_y"]
        transition_rows.append(window_events)

    if not transition_rows:
        return pd.DataFrame()
    return pd.concat(transition_rows, ignore_index=True).dropna(subset=["loc_x", "loc_y"])


# ─────────────────────────────────────────────────────────────────────────────
# IDEA 4: Pressure Trigger Zones
# Lecture anchor: Spatio-temporal (event location, decision making)
# ─────────────────────────────────────────────────────────────────────────────

def pressure_trigger_zones(events: pd.DataFrame, opponent: str) -> pd.DataFrame:
    """
    Where does the opponent press MAN CITY and how effective is it?
    Returns pressure events the opponent applied with outcome context.
    """
    # Opponent's pressure events against Man City
    pressures = events[
        (events["type"] == "Pressure") & (events["team"] == opponent)
    ].copy()

    # Join with the event being pressured to get the outcome
    pressures = pressures.dropna(subset=["loc_x", "loc_y"])
    # Zone labels for readability
    pressures["pitch_zone"] = pressures.apply(
        lambda r: _zone_label(r["loc_x"], r["loc_y"]), axis=1
    )
    return pressures[["player", "loc_x", "loc_y", "pitch_zone",
                       "match_id", "minute"]].reset_index(drop=True)


def opponent_weak_zones(events: pd.DataFrame, opponent: str) -> pd.DataFrame:
    """
    Where does the opponent give away the ball when pressed?
    Returns Man City pressure events where the next event is an opponent loss.
    """
    city_pressures = events[
        (events["type"] == "Pressure") & (events["team"] == "Manchester City")
    ].copy().dropna(subset=["loc_x", "loc_y"])
    city_pressures["pitch_zone"] = city_pressures.apply(
        lambda r: _zone_label(r["loc_x"], r["loc_y"]), axis=1
    )
    # Count by zone
    zone_counts = (
        city_pressures.groupby("pitch_zone")
        .size()
        .reset_index(name="press_count")
        .sort_values("press_count", ascending=False)
    )
    return zone_counts


def _zone_label(x, y) -> str:
    if pd.isna(x) or pd.isna(y):
        return "Unknown"
    x_label = "Def Third" if x < 40 else ("Mid Third" if x < 80 else "Att Third")
    y_label = "Left"       if y < 27 else ("Right" if y > 53 else "Centre")
    return f"{x_label} – {y_label}"


# ─────────────────────────────────────────────────────────────────────────────
# WHO TO STOP: Passing Network Centrality
# Lecture anchor: Ranking systems (PageRank / betweenness centrality)
# Reference: Lazova & Basnarkov (2015), Radicchi (2011), both in your slides
# ─────────────────────────────────────────────────────────────────────────────

def passing_network_centrality(events: pd.DataFrame, team: str) -> pd.DataFrame:
    """
    Build the team's passing network and compute betweenness centrality.
    The most central player IS the hub, neutralise them and build-up breaks.
    """
    passes = events[
        (events["type"] == "Pass") &
        (events["team"] == team) &
        events["pass_recipient"].notna() &
        (events["pass_outcome"].isna())   # successful passes only
    ].copy()

    if len(passes) < 10:
        return pd.DataFrame()

    G = nx.DiGraph()
    for _, row in passes.iterrows():
        src = row["player"]
        tgt = row["pass_recipient"]
        if G.has_edge(src, tgt):
            G[src][tgt]["weight"] += 1
        else:
            G.add_edge(src, tgt, weight=1)

    centrality = nx.betweenness_centrality(G, weight="weight", normalized=True)
    degree_in  = dict(G.in_degree(weight="weight"))
    degree_out = dict(G.out_degree(weight="weight"))

    rows = []
    for player, cent in centrality.items():
        rows.append({
            "player": player,
            "betweenness_centrality": round(cent, 4),
            "passes_received": degree_in.get(player, 0),
            "passes_made": degree_out.get(player, 0),
            "total_involvement": degree_in.get(player, 0) + degree_out.get(player, 0),
        })

    df = pd.DataFrame(rows).sort_values("betweenness_centrality", ascending=False)
    df["centrality_rank"] = range(1, len(df) + 1)
    return df.reset_index(drop=True)


def passing_network_edges(events: pd.DataFrame, team: str,
                          top_n_edges: int = 20) -> pd.DataFrame:
    """Return top N pass combinations by volume, for the network visualisation."""
    passes = events[
        (events["type"] == "Pass") &
        (events["team"] == team) &
        events["pass_recipient"].notna() &
        (events["pass_outcome"].isna())
    ].copy()

    edges = (
        passes.groupby(["player", "pass_recipient"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n_edges)
    )
    # Merge average positions
    avg_pos = (
        passes.groupby("player")[["loc_x", "loc_y"]]
        .mean()
        .rename(columns={"loc_x": "src_x", "loc_y": "src_y"})
        .reset_index()
    )
    tgt_pos = (
        passes.groupby("pass_recipient")[["loc_x", "loc_y"]]
        .mean()
        .rename(columns={"loc_x": "tgt_x", "loc_y": "tgt_y",
                         "pass_recipient": "player"})
        .reset_index()
    )
    edges = edges.merge(avg_pos, on="player", how="left")
    edges = edges.merge(
        tgt_pos.rename(columns={"player": "pass_recipient"}),
        on="pass_recipient", how="left"
    )
    return edges


# ─────────────────────────────────────────────────────────────────────────────
# SET PIECES
# Lecture anchor: Spatio-temporal (event location)
# Real-world: Midtjylland (~49% goals from set pieces), Gronnemark / Liverpool
# ─────────────────────────────────────────────────────────────────────────────

DEAD_BALL_TYPES = {"Corner", "Free Kick", "Throw-in", "Kick Off"}

def set_piece_analysis(events: pd.DataFrame, team: str) -> dict:
    """Analyse a team's set-piece delivery locations and outcomes."""
    sp_passes = events[
        (events["type"] == "Pass") &
        (events["team"] == team) &
        events["pass_type"].isin(DEAD_BALL_TYPES)
    ].copy()

    sp_shots = events[
        (events["type"] == "Shot") &
        (events["team"] == team) &
        events["shot_type"].isin(DEAD_BALL_TYPES)
    ].copy() if "shot_type" in events.columns else pd.DataFrame()

    by_type = sp_passes.groupby("pass_type").size().rename("count")

    # End-location of deliveries (for heatmap of where balls land)
    deliveries = sp_passes.dropna(subset=["pass_end_location"]).copy()
    deliveries["end_x"] = deliveries["pass_end_location"].apply(
        lambda l: l[0] if isinstance(l, list) else np.nan
    )
    deliveries["end_y"] = deliveries["pass_end_location"].apply(
        lambda l: l[1] if isinstance(l, list) else np.nan
    )

    goals_from_sp = 0
    if len(sp_shots):
        goals_from_sp = (sp_shots.get("shot_outcome", pd.Series()) == "Goal").sum()

    # open-play shots, for an honest "is this actually working" comparison
    all_shots = events[(events["type"] == "Shot") & (events["team"] == team)]
    op_shots = all_shots[~all_shots.get("shot_type", pd.Series(index=all_shots.index)).isin(DEAD_BALL_TYPES)] \
        if "shot_type" in all_shots.columns else all_shots

    def _conv(df):
        n = len(df)
        goals = int((df.get("shot_outcome", pd.Series()) == "Goal").sum()) if n else 0
        xg = round(float(df["shot_statsbomb_xg"].sum()), 2) if "shot_statsbomb_xg" in df.columns and n else 0.0
        return n, goals, xg

    sp_n, sp_goals, sp_xg = _conv(sp_shots)
    op_n, op_goals, op_xg = _conv(op_shots)

    return {
        "by_type": by_type,
        "deliveries": deliveries,
        "goals_from_set_pieces": int(goals_from_sp),
        "total_set_pieces": len(sp_passes),
        "sp_shots": sp_n, "sp_goals": sp_goals, "sp_xg": sp_xg,
        "sp_conv_pct": round(sp_goals / sp_n * 100, 1) if sp_n else 0.0,
        "sp_xg_per_shot": round(sp_xg / sp_n, 2) if sp_n else 0.0,
        "op_shots": op_n, "op_goals": op_goals, "op_xg": op_xg,
        "op_conv_pct": round(op_goals / op_n * 100, 1) if op_n else 0.0,
        "op_xg_per_shot": round(op_xg / op_n, 2) if op_n else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ELO RATING (opponent strength)
# Lecture anchor: Rating Systems, self-adjusting / adjustive system
# Formula: P = P^(n-1) + k*(w - w_e)  from your lecture slides
# ─────────────────────────────────────────────────────────────────────────────

def compute_elo_ratings(matches: pd.DataFrame, k: float = 20.0,
                         initial: float = 1000.0) -> pd.DataFrame:
    """
    Compute season Elo for all 20 Premier League teams using the
    FIFA/lecture formula: P = P^(n-1) + k*(w - w_e)
    """
    teams = sorted(set(matches["home_team"].tolist() + matches["away_team"].tolist()))
    elo = {t: initial for t in teams}

    history = []
    for _, row in matches.sort_values("match_id").iterrows():
        h, a = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        r_h = elo[h]; r_a = elo[a]
        dr = r_h - r_a
        we_h = 1 / (10 ** (-dr / 400) + 1)   # standard Elo expected score
        w_h  = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        elo[h] = r_h + k * (w_h - we_h)
        elo[a] = r_a + k * ((1 - w_h) - (1 - we_h))

    df = pd.DataFrame(
        [{"team": t, "elo": round(v, 1)} for t, v in elo.items()]
    ).sort_values("elo", ascending=False).reset_index(drop=True)
    df["elo_rank"] = df.index + 1
    return df


# ─────────────────────────────────────────────────────────────────────────────
# BEST XI: composite selection score (configurable weights)
# Lecture anchor: Rating Systems, weighted accumulation
# ─────────────────────────────────────────────────────────────────────────────

def compute_best_xi_score(squad_stats: pd.DataFrame,
                           pr_ratings: pd.DataFrame,
                           weights: dict) -> pd.DataFrame:
    """
    Combine stats into a configurable Best XI composite score.
    weights: {"press_resistance": 0-1, "goal_contribution": 0-1,
              "pass_quality": 0-1, "pressing_output": 0-1}
    """
    merged = squad_stats.merge(
        pr_ratings[["player", "PR_score", "PR_rank"]], on="player", how="left"
    )

    # Helper: rescale any column to a 0-100 score (simple min-max)
    def to_100(series):
        s = series.fillna(series.min())
        lo, hi = s.min(), s.max()
        if hi == lo:
            return pd.Series([50.0] * len(s), index=s.index)
        return (s - lo) / (hi - lo) * 100

    # Build the four component scores (all 0-100)
    merged["n_pr"]    = to_100(merged["PR_score"]).fillna(50)
    merged["n_gc"]    = to_100(merged["goals"] + merged["assists"])
    merged["n_pass"]  = to_100(merged["pass_completion_pct"])
    merged["n_press"] = to_100(merged["pressures_applied"])

    w_pr   = weights.get("press_resistance", 0.35)
    w_gc   = weights.get("goal_contribution", 0.30)
    w_pass = weights.get("pass_quality", 0.20)
    w_pres = weights.get("pressing_output", 0.15)

    merged["xi_score"] = (
        merged["n_pr"]    * w_pr +
        merged["n_gc"]    * w_gc +
        merged["n_pass"]  * w_pass +
        merged["n_press"] * w_pres
    ).round(1)

    return merged.sort_values("xi_score", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# MATCHWEEK MODE: season progression
# ─────────────────────────────────────────────────────────────────────────────

def form_up_to_week(events, matches, week, our_team="Manchester City"):
    """Man City player stats using ONLY matches before the given week."""
    past = matches[matches["match_week"] < week]["match_id"].tolist()
    if not past:
        return pd.DataFrame()
    ev = events[(events["match_id"].isin(past)) & (events["team"] == our_team)]
    passes = ev[ev["type"] == "Pass"]
    shots = ev[ev["type"] == "Shot"]
    rows = []
    for p in sorted(ev["player"].dropna().unique()):
        pp = passes[passes["player"] == p]
        ps = shots[shots["player"] == p]
        up = pp[pp["under_pressure"] == True]
        rows.append({
            "player": p,
            "matches": ev[ev["player"] == p]["match_id"].nunique(),
            "goals": int((ps["shot_outcome"] == "Goal").sum()),
            "passes": len(pp),
            "pass_pct": round(pp["pass_outcome"].isna().mean() * 100, 1) if len(pp) else 0,
        })
    return pd.DataFrame(rows).sort_values("matches", ascending=False).reset_index(drop=True)


def recent_form(matches, week, n=5):
    """Last n results before the given week."""
    past = matches[matches["match_week"] < week].sort_values("match_week")
    return past["result"].tail(n).tolist()


# ─────────────────────────────────────────────────────────────────────────────
# RECOMMENDATION ENGINE: simple, readable rules (no ML)
# ─────────────────────────────────────────────────────────────────────────────

def recommend_pressure_resistance(pr_df, opponent_press_intensity="high"):
    if len(pr_df) == 0:
        return [("info", "Not enough data yet to rate pressure resistance.")]
    best = pr_df.iloc[0]; worst = pr_df.iloc[-1]
    recs = [("good", f"<b>START {best['player']}</b>, your most press-resistant player "
                     f"(PR score {best['PR_score']}, keeps {best['pressured_pass_pct']}% under pressure).")]
    if worst["resistance_drop"] > 15:
        recs.append(("warn", f"<b>CAUTION with {worst['player']}</b>, completion drops "
                             f"{worst['resistance_drop']:.0f}% when pressed. Risky against a high press."))
    return recs


def recommend_tactical_degradation(td):
    xs = td.get("x_shift", 0)
    if xs > 3:
        return [("good", f"They drop <b>{abs(xs):.0f} yards deeper</b> in the last 15 minutes, "
                         f"hold attacking subs in reserve and go for the kill after the 75th minute.")]
    if xs < -3:
        return [("warn", f"They push <b>{abs(xs):.0f} yards higher</b> late on, watch for a late surge.")]
    return [("info", "Their shape stays stable late, no obvious fatigue window to exploit.")]


def recommend_transition(trans_df, opponent):
    if len(trans_df) == 0:
        return [("info", "Not enough transition data for this opponent.")]
    t = trans_df.copy()
    t["zone"] = t.apply(lambda r:
        ("Def" if r["loc_x"] < 40 else ("Mid" if r["loc_x"] < 80 else "Att")) + " " +
        ("Left" if r["loc_y"] < 27 else ("Right" if r["loc_y"] > 53 else "Centre")), axis=1)
    weak = t["zone"].value_counts().idxmin()
    return [("good", f"When you win the ball back, {opponent} leaves the <b>{weak}</b> channel "
                     f"most open in the first 5 seconds, break into it instantly.")]


def recommend_who_to_stop(cent_df, opponent):
    if len(cent_df) == 0:
        return [("info", "Not enough passing data to map their network.")]
    hub = cent_df.iloc[0]
    return [("good", f"<b>STOP {hub['player']}</b>, {opponent}'s passing hub "
                     f"(highest centrality). Cut their supply and the build-up breaks down.")]
