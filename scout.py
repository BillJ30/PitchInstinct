"""
PitchInstinct: Scouting Engine
Turns an opponent's real match data into plain-English scouting verdicts.
Everything here is descriptive (counts, averages, shares). No ML.

The flow:  scout_opponent() -> a profile dict
           battlegrounds()  -> the key threats/opportunities (text)
           recommend_xi()    -> City XI chosen to BEAT this opponent, with reasons
"""

import pandas as pd
import numpy as np
import networkx as nx


# ── thresholds (calibrated from the data) ─────────────────────────────────────
PRESS_HIGH = 150     # pressures per match above this = aggressive press
PRESS_LOW  = 110     # below this = passive
FLANK_BIAS = 0.45    # if one flank has >45% of final-third passes = clear bias


def _per_match(events, team):
    n = events["match_id"].nunique()
    return max(n, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 1. SCOUT THE OPPONENT  → a profile we can both display and reason from
# ═════════════════════════════════════════════════════════════════════════════

def scout_opponent(events, opponent):
    """events = full-season events for the opponent's matches (both teams)."""
    nm = _per_match(events, opponent)
    own = events[events["team"] == opponent]

    # --- pressing intensity ---
    pressures = own[own["type"] == "Pressure"].dropna(subset=["loc_x", "loc_y"])
    press_pm = len(pressures) / nm
    if press_pm >= PRESS_HIGH:
        press_level, press_word = "High", "an aggressive high press"
    elif press_pm <= PRESS_LOW:
        press_level, press_word = "Low", "a passive, sit-back block"
    else:
        press_level, press_word = "Medium", "a balanced pressing approach"

    # --- attacking flank bias (final-third passes) ---
    ft = own[(own["type"] == "Pass") & (own["loc_x"] > 80)].dropna(subset=["loc_y"])
    tot = max(len(ft), 1)
    left  = (ft["loc_y"] < 26.7).sum() / tot
    right = (ft["loc_y"] > 53.3).sum() / tot
    if left >= FLANK_BIAS and left > right:
        flank, flank_word = "Left", "down their left side"
    elif right >= FLANK_BIAS and right > left:
        flank, flank_word = "Right", "down their right side"
    else:
        flank, flank_word = "Central", "through central areas"

    # --- attacking output ---
    shots = own[own["type"] == "Shot"]
    xg_pm = shots["shot_statsbomb_xg"].sum() / nm if "shot_statsbomb_xg" in shots.columns else 0
    shots_pm = len(shots) / nm
    potent = "potent" if xg_pm >= 1.4 else ("blunt" if xg_pm < 1.0 else "steady")

    # --- leakiness (shots conceded = the OTHER team's shots in their matches) ---
    conceded = events[(events["type"] == "Shot") & (events["team"] != opponent)].copy()
    conc_pm = len(conceded) / nm
    goals_conc_pm = (conceded["shot_outcome"] == "Goal").sum() / nm
    leaky = "leaky" if goals_conc_pm >= 1.3 else ("solid" if goals_conc_pm < 0.9 else "average")
    # flip conceded shots into the opponent's defensive half (near their own goal)
    conceded["def_x"] = 120 - conceded["loc_x"]
    conceded["def_y"] = 80 - conceded["loc_y"]

    # --- formation (most common starting shape) ---
    sxi = events[(events["type"] == "Starting XI") & (events["team"] == opponent)]
    formations = []
    for _, r in sxi.iterrows():
        t = r["tactics"]
        if isinstance(t, dict) and t.get("formation"):
            formations.append(str(int(t["formation"])))
    formation = pd.Series(formations).mode()[0] if formations else "?"

    return {
        "opponent": opponent,
        "matches": nm,
        "press_pm": round(press_pm),
        "press_level": press_level,
        "press_word": press_word,
        "flank": flank,
        "flank_word": flank_word,
        "flank_left_pct": round(left * 100),
        "flank_right_pct": round(right * 100),
        "xg_pm": round(xg_pm, 2),
        "shots_pm": round(shots_pm, 1),
        "potent": potent,
        "conc_pm": round(conc_pm, 1),
        "goals_conc_pm": round(goals_conc_pm, 2),
        "leaky": leaky,
        "formation": formation,
        # raw data for the visuals
        "pressure_points": pressures[["loc_x", "loc_y"]],
        "shot_points": shots.dropna(subset=["loc_x", "loc_y"])[["loc_x", "loc_y"]],
        "conceded_points": conceded.dropna(subset=["def_x", "def_y"])[["def_x", "def_y"]],
    }


def scout_key_player(events, opponent):
    """Most central OUTFIELD player in the opponent's passing network (who to stop)."""
    # find goalkeepers to exclude (they always rank high on centrality)
    keepers = set(
        events[(events["team"] == opponent) &
               (events.get("position", "") == "Goalkeeper")]["player"].dropna()
    )
    passes = events[(events["type"] == "Pass") & (events["team"] == opponent) &
                    events["pass_recipient"].notna() &
                    (events["pass_outcome"].isna())]
    if len(passes) < 20:
        return None
    G = nx.DiGraph()
    for _, r in passes.iterrows():
        a, b = r["player"], r["pass_recipient"]
        if G.has_edge(a, b): G[a][b]["weight"] += 1
        else: G.add_edge(a, b, weight=1)
    cent = nx.betweenness_centrality(G, weight="weight", normalized=True)
    # drop keepers
    cent = {p: c for p, c in cent.items() if p not in keepers}
    if not cent:
        return None
    top = max(cent, key=cent.get)
    return {"player": top, "centrality": round(cent[top], 3)}


def scout_fatigue(events, opponent):
    """Do they drop deeper in the last 15 minutes?"""
    dtypes = ["Pressure", "Tackle", "Interception", "Clearance", "Block"]
    d = events[(events["type"].isin(dtypes)) & (events["team"] == opponent)].dropna(subset=["loc_x", "minute"])
    early = d[d["minute"] <= 75]["loc_x"].mean()
    late  = d[d["minute"] > 75]["loc_x"].mean()
    if pd.isna(early) or pd.isna(late):
        return {"shift": 0.0, "fades": False}
    shift = round(late - early, 1)
    return {"shift": shift, "fades": shift < -3}  # more negative x = deeper


# ═════════════════════════════════════════════════════════════════════════════
# 2. THE BATTLEGROUNDS  → the 3-4 things that will decide this game
# ═════════════════════════════════════════════════════════════════════════════

def battlegrounds(profile, key_player, fatigue):
    out = []
    # threat: their press
    if profile["press_level"] == "High":
        out.append(("threat", f"{profile['opponent']} run {profile['press_word']} "
                              f"({profile['press_pm']} pressures/game). Your build-up will be "
                              f"hurried, you need players who keep the ball when pressed."))
    elif profile["press_level"] == "Low":
        out.append(("opportunity", f"They sit back ({profile['press_pm']} pressures/game), "
                                   f"you'll have time on the ball. Pick your best passers to break them down."))
    # threat: their attacking side
    if profile["flank"] != "Central":
        out.append(("threat", f"They attack mostly {profile['flank_word']} "
                              f"({profile['flank_'+profile['flank'].lower()+'_pct']}% of final-third passes). "
                              f"Make sure that flank is defensively solid."))
    # who to stop
    if key_player:
        out.append(("threat", f"Their build-up runs through {key_player['player']}, "
                              f"assign someone to cut his supply and they struggle to play out."))
    # opportunity: leaky
    if profile["leaky"] == "leaky":
        out.append(("opportunity", f"They are defensively leaky ({profile['goals_conc_pm']} goals "
                                   f"conceded/game), your front line will get chances."))
    elif profile["leaky"] == "solid":
        out.append(("threat", f"They defend well ({profile['goals_conc_pm']} goals conceded/game), "
                              f"chances will be rare, so take them."))
    # opportunity: fatigue
    if fatigue["fades"]:
        out.append(("opportunity", f"They drop {abs(fatigue['shift']):.0f} yards deeper after 75 minutes, "
                                   f"keep pace on the bench and go for the kill late."))
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3. RECOMMEND CITY'S XI  → chosen to BEAT this specific opponent, with reasons
# ═════════════════════════════════════════════════════════════════════════════

def opponent_priorities(profile):
    """Turn the scout profile into selection priorities (weights + a why)."""
    # base weights
    w = {"press_resistance": 0.30, "goal_threat": 0.30,
         "passing": 0.20, "pressing": 0.20}
    reasons = []

    if profile["press_level"] == "High":
        w["press_resistance"] += 0.20
        reasons.append(f"they press high, so **press resistance** is prioritised")
    if profile["press_level"] == "Low":
        w["passing"] += 0.15
        reasons.append(f"they sit deep, so **passing quality** to unlock them is prioritised")
    if profile["leaky"] == "leaky":
        w["goal_threat"] += 0.15
        reasons.append(f"they are leaky, so **goal threat** is prioritised")
    if profile["leaky"] == "solid":
        w["goal_threat"] += 0.10
        reasons.append(f"they defend well, so clinical **goal threat** still matters")

    # normalise so weights sum to 1
    total = sum(w.values())
    w = {k: round(v / total, 3) for k, v in w.items()}
    return w, reasons


def recommend_xi(squad_scored, positions, profile, key_player, formation=None):
    """
    squad_scored : the City squad already scored with opponent-driven weights.
    formation    : dict like {"GK":1,"DEF":4,"MID":3,"FWD":3}. Defaults to 4-3-3.
    Returns the XI with a per-player REASON tied to the opponent.
    """
    if formation is None:
        formation = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}

    # only merge positions if they aren't already on the dataframe
    if "pos_group" in squad_scored.columns:
        df = squad_scored.copy()
    else:
        df = squad_scored.merge(positions[["player", "pos_group", "position"]],
                                on="player", how="left")
    df["pos_group"] = df["pos_group"].fillna("MID")

    picks = []
    for grp, n in formation.items():
        for _, row in df[df["pos_group"] == grp].head(n).iterrows():
            picks.append(row)
    xi = pd.DataFrame(picks)

    # add a reason per player based on what this opponent demands
    def reason(row):
        if profile["press_level"] == "High" and row.get("PR_score", 0) >= 70:
            return "Press-resistant, keeps the ball vs their high press"
        if profile["leaky"] == "leaky" and (row.get("goals", 0) + row.get("assists", 0)) >= 5:
            return "Goal threat, to punish their leaky defence"
        if row["pos_group"] == "MID" and row.get("pressures_applied", 0) > df["pressures_applied"].median():
            return "Hard presser, can press their build-up"
        if row.get("PR_score", 0) >= 70:
            return "Reliable on the ball under pressure"
        return "Top composite score for this matchup"

    xi["reason"] = xi.apply(reason, axis=1)
    return xi
