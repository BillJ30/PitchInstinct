"""
PitchInstinct: Feature engine (descriptive only, no ML).
Opposition threat assessment, matchup analysis, team balance, roles, duels.
Kept readable: simple groupbys, counts, shares, and one small Poisson model.
"""

import pandas as pd
import numpy as np
import math

SET_PIECE_TYPES = {"Corner", "Free Kick"}
LEFT_MAX, RIGHT_MIN = 26.7, 53.3   # pitch thirds on the 80-wide axis


# ════════════════════════════════════════════════════════════════════════════
# OPPOSITION THREAT ASSESSMENT
# ════════════════════════════════════════════════════════════════════════════

def danger_men(events, opponent, top_n=5):
    """Top attacking threats by goals, xG, assists, key passes."""
    own = events[events["team"] == opponent]
    shots = own[own["type"] == "Shot"]
    passes = own[own["type"] == "Pass"]

    rows = []
    for p in own["player"].dropna().unique():
        ps = shots[shots["player"] == p]
        pp = passes[passes["player"] == p]
        goals = int((ps["shot_outcome"] == "Goal").sum())
        xg = float(ps["shot_statsbomb_xg"].sum()) if "shot_statsbomb_xg" in ps.columns else 0
        assists = int(pp["pass_goal_assist"].sum()) if "pass_goal_assist" in pp.columns else 0
        keyp = int(pp["pass_shot_assist"].sum()) if "pass_shot_assist" in pp.columns else 0
        threat = goals + assists
        if threat == 0 and xg < 1:
            continue
        rows.append({"player": p, "goals": goals, "xg": round(xg, 1),
                     "assists": assists, "key_passes": keyp,
                     "threat": threat})
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["threat", "xg"], ascending=False).head(top_n).reset_index(drop=True)


def likely_xi(events, opponent):
    """Their most-used players (by starts) with positions, the expected XI."""
    sxi = events[(events["type"] == "Starting XI") & (events["team"] == opponent)]
    counts = {}
    for _, r in sxi.iterrows():
        t = r["tactics"]
        if isinstance(t, dict):
            for pl in t.get("lineup", []):
                name = pl.get("player", {}).get("name")
                pos = pl.get("position", {}).get("name", "")
                if name:
                    counts.setdefault(name, {"starts": 0, "position": pos})
                    counts[name]["starts"] += 1
    rows = [{"player": k, **v} for k, v in counts.items()]
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values("starts", ascending=False).head(11).reset_index(drop=True)


def set_piece_profile(events, opponent):
    """Their set-piece threat (goals scored) and vulnerability (goals conceded)."""
    own = events[events["team"] == opponent]
    opp = events[events["team"] != opponent]

    def sp_shots(df):
        if "shot_type" not in df.columns:
            return df.iloc[0:0]
        return df[(df["type"] == "Shot") & (df["shot_type"].isin(SET_PIECE_TYPES))]

    own_sp = sp_shots(own)
    conc_sp = sp_shots(opp)
    nm = max(events["match_id"].nunique(), 1)
    return {
        "threat_shots": len(own_sp),
        "threat_goals": int((own_sp["shot_outcome"] == "Goal").sum()) if len(own_sp) else 0,
        "vuln_shots": len(conc_sp),
        "vuln_goals": int((conc_sp["shot_outcome"] == "Goal").sum()) if len(conc_sp) else 0,
        "threat_pg": round(len(own_sp) / nm, 1),
        "vuln_pg": round(len(conc_sp) / nm, 1),
    }


def pressing_triggers(events, opponent):
    """Where they WIN the ball (recoveries) and LOSE it (turnovers), by third."""
    own = events[events["team"] == opponent]

    wins = own[own["type"].isin(["Ball Recovery", "Interception"])].dropna(subset=["loc_x"])
    loses = own[own["type"].isin(["Dispossessed", "Miscontrol"])].dropna(subset=["loc_x"])

    def third(x):
        return "Defensive" if x < 40 else ("Middle" if x < 80 else "Attacking")

    win_third = wins["loc_x"].apply(third).value_counts(normalize=True) if len(wins) else pd.Series()
    lose_third = loses["loc_x"].apply(third).value_counts(normalize=True) if len(loses) else pd.Series()

    # verdict: where do they win it most?
    win_zone = win_third.idxmax() if len(win_third) else "Middle"
    return {
        "win_points": wins[["loc_x", "loc_y"]],
        "lose_points": loses[["loc_x", "loc_y"]],
        "win_third": win_third, "lose_third": lose_third,
        "win_zone": win_zone,
    }


def _match_score_timeline(match_events):
    """Return sorted list of (minute, scoring_team) goals in a match."""
    goals = match_events[(match_events["type"] == "Shot") &
                         (match_events["shot_outcome"] == "Goal")]
    timeline = [(int(r["minute"]), r["team"]) for _, r in goals.iterrows()]
    # own goals count for the other team
    og = match_events[match_events["type"] == "Own Goal Against"]
    return sorted(timeline, key=lambda x: x[0])


def game_state_tendencies(events, opponent):
    """How their pressing intensity changes when leading / level / trailing."""
    states = {"Leading": {"min": 0, "press": 0},
              "Level":   {"min": 0, "press": 0},
              "Trailing":{"min": 0, "press": 0}}

    for mid, mev in events.groupby("match_id"):
        timeline = _match_score_timeline(mev)
        # build score difference (opponent perspective) minute by minute
        opp_goals, other_goals = 0, 0
        segments = []  # (start, end, state)
        last_min = 0
        for minute, team in timeline:
            state = ("Leading" if opp_goals > other_goals else
                     "Trailing" if opp_goals < other_goals else "Level")
            segments.append((last_min, minute, state))
            if team == opponent: opp_goals += 1
            else: other_goals += 1
            last_min = minute
        # final segment to end (~95)
        state = ("Leading" if opp_goals > other_goals else
                 "Trailing" if opp_goals < other_goals else "Level")
        segments.append((last_min, 95, state))

        pressures = mev[(mev["type"] == "Pressure") & (mev["team"] == opponent)]
        for start, end, state in segments:
            states[state]["min"] += max(end - start, 0)
            states[state]["press"] += len(pressures[(pressures["minute"] >= start) &
                                                    (pressures["minute"] < end)])

    out = {}
    for s, v in states.items():
        out[s] = round(v["press"] / v["min"], 2) if v["min"] > 0 else 0.0  # pressures/min
    return out


# ════════════════════════════════════════════════════════════════════════════
# THE X-FACTORS: what will decide this game (your "blitz" idea)
# ════════════════════════════════════════════════════════════════════════════

def x_factors(profile, sp, key_player, gs, fatigue):
    """Surface the 2-3 decisive levers, each with a quantified edge."""
    factors = []

    # set-piece swing
    if sp["threat_pg"] >= 0.4 or sp["threat_goals"] >= 6:
        factors.append(("danger", "Set-piece threat",
            f"They generate {sp['threat_pg']} set-piece shots/game ({sp['threat_goals']} goals). "
            f"Win the first contact or this swings the game."))
    if sp["vuln_pg"] >= 0.4:
        factors.append(("edge", "Set-piece weakness",
            f"They concede {sp['vuln_pg']} set-piece shots/game, load the box on your dead balls."))

    # game-state behaviour
    if gs.get("Trailing", 0) > gs.get("Leading", 0) * 1.4 and gs.get("Leading", 0) > 0:
        factors.append(("edge", "They chase recklessly",
            f"When behind they press {gs['Trailing']}/min vs {gs['Leading']}/min when ahead, "
            f"score first and space opens up behind them."))
    elif gs.get("Leading", 0) and gs.get("Leading", 0) < gs.get("Level", 99):
        factors.append(("danger", "They game-manage a lead",
            f"They drop off when ahead ({gs['Leading']}/min), don't concede first."))

    # key player dependency
    if key_player:
        factors.append(("danger", f"Stop {key_player['player']}",
            f"Their build-up runs through him (centrality {key_player['centrality']}). "
            f"Nullify him and their attack stalls."))

    # late fatigue
    if fatigue.get("fades"):
        factors.append(("edge", "They fade late",
            f"They drop {abs(fatigue['shift']):.0f} yards deeper after 75', keep pace on the bench."))

    # press as a battleground
    if profile["press_level"] == "High":
        factors.append(("danger", "Their high press",
            f"{profile['press_pm']} pressures/game, beat the first line and you're through."))

    return factors[:4]


# ════════════════════════════════════════════════════════════════════════════
# MATCHUP: your team vs theirs
# ════════════════════════════════════════════════════════════════════════════

def formation_xy(formation_str, side="home"):
    """
    Turn a formation string (e.g. '433','4231') into (x,y) points on a
    horizontal statsbomb pitch (120 long x 80 wide).
    side 'home' = your team on the left attacking right (x 6->54).
    side 'away' = opponent on the right attacking left (x 114->66).
    """
    digits = [int(d) for d in str(formation_str) if d.isdigit()]
    if not digits:
        digits = [4, 3, 3]
    lines = [("GK", 1)] + [(str(d), d) for d in digits]
    n = len(lines)
    pts = []
    for li, (lab, count) in enumerate(lines):
        frac = li / (n - 1) if n > 1 else 0
        if side == "home":
            x = 6 + frac * 48          # 6 .. 54
        else:
            x = 114 - frac * 48        # 114 .. 66
        ys = np.linspace(14, 66, count + 2)[1:-1]
        for y in ys:
            pts.append((x, y, lab))
    return pts


def formation_mismatches(your_form, opp_form):
    """Compare the two shapes and flag where one side overloads."""
    def lines(f):
        return [int(d) for d in str(f) if d.isdigit()]
    yl, ol = lines(your_form), lines(opp_form)
    out = []
    # central midfield overload: compare middle line sizes
    your_mid = yl[1] if len(yl) > 1 else 3
    opp_mid = ol[1] if len(ol) > 1 else 3
    if opp_mid > your_mid:
        out.append(("warn", f"They field {opp_mid} in midfield vs your {your_mid}, "
                            f"central overload risk. Drop a man in or stay compact."))
    elif your_mid > opp_mid:
        out.append(("good", f"You have {your_mid} in midfield vs their {opp_mid}, "
                           f"control the centre and dictate tempo."))
    # forwards vs defenders
    opp_fwd = ol[-1] if ol else 3
    your_def = yl[0] if yl else 4
    if opp_fwd >= your_def:
        out.append(("warn", f"Their {opp_fwd} forwards stretch your {your_def} defenders, "
                            f"watch the channels, consider an extra centre-back."))
    else:
        out.append(("good", f"Your {your_def} defenders outnumber their {opp_fwd} forwards, "
                           f"a spare man to step out and build."))
    return out


def key_duels(dmen, opp_events, opponent, your_xi, positions):
    """For each danger man, find his channel and your defender there."""
    own = opp_events[opp_events["team"] == opponent]
    duels = []
    your_def = your_xi[your_xi["pos_group"] == "DEF"]
    for _, d in dmen.head(3).iterrows():
        touches = own[own["player"] == d["player"]].dropna(subset=["loc_y"])
        if len(touches) == 0:
            continue
        avg_y = touches["loc_y"].mean()
        # their attacking channel -> the opposite side of your defence
        if avg_y < LEFT_MAX:
            channel, opp_side = "their left", "your right"
        elif avg_y > RIGHT_MIN:
            channel, opp_side = "their right", "your left"
        else:
            channel, opp_side = "central", "your centre-backs"
        # pick your defender on that side (simple split by index)
        marker = None
        if len(your_def):
            if "right" in opp_side:
                marker = your_def.iloc[-1]["player"]
            elif "left" in opp_side:
                marker = your_def.iloc[0]["player"]
            else:
                mid = len(your_def) // 2
                marker = your_def.iloc[mid]["player"]
        duels.append({"threat": d["player"], "channel": channel,
                      "goals": d["goals"], "assists": d["assists"],
                      "marker": marker, "your_side": opp_side})
    return duels


# ════════════════════════════════════════════════════════════════════════════
# PROJECTED SCORE: descriptive xG blend + Poisson (no ML)
# ════════════════════════════════════════════════════════════════════════════

def _poisson(k, lam):
    return (lam ** k) * math.exp(-lam) / math.factorial(k)


def projected_score(city_events, city_matches, opp_events, opponent, risk=0.0):
    """Blend each side's attack and the other's defence, then Poisson for W/D/L.
    risk: -1 (defensive) .. 0 (balanced) .. +1 (attacking), a more attacking approach
    creates more chances for both sides, a defensive one fewer for both."""
    nmc = max(city_matches["match_id"].nunique(), 1)
    city_for = city_matches["city_score"].mean()
    city_against = city_matches["opp_score"].mean()

    nmo = max(opp_events["match_id"].nunique(), 1)
    own = opp_events[opp_events["team"] == opponent]
    opp_for = (own[own["type"] == "Shot"]["shot_outcome"] == "Goal").sum() / nmo
    conc = opp_events[(opp_events["type"] == "Shot") & (opp_events["team"] != opponent)]
    opp_against = (conc["shot_outcome"] == "Goal").sum() / nmo

    exp_city = (city_for + opp_against) / 2
    exp_opp = (opp_for + city_against) / 2

    # attacking approach: push City's own rate up more than the opponent's; defensive: the reverse
    exp_city *= 1 + 0.18 * risk
    exp_opp *= 1 + 0.10 * risk
    exp_city = max(exp_city, 0.1); exp_opp = max(exp_opp, 0.1)

    # Poisson grid 0..8
    p_city_win = p_draw = p_opp_win = 0.0
    for a in range(9):
        for b in range(9):
            p = _poisson(a, exp_city) * _poisson(b, exp_opp)
            if a > b: p_city_win += p
            elif a == b: p_draw += p
            else: p_opp_win += p
    return {
        "exp_city": round(exp_city, 2), "exp_opp": round(exp_opp, 2),
        "p_win": round(p_city_win * 100), "p_draw": round(p_draw * 100),
        "p_lose": round(p_opp_win * 100),
        "scoreline": f"{round(exp_city)}-{round(exp_opp)}",
    }


# ════════════════════════════════════════════════════════════════════════════
# TEAM BALANCE, ROLES, PLAYER COMPARISON
# ════════════════════════════════════════════════════════════════════════════

RADAR_AXES = ["Press Resistance", "Goal Threat", "Creativity", "Passing", "Work-rate"]

def _norm(series):
    s = series.fillna(series.min())
    lo, hi = s.min(), s.max()
    if hi == lo:
        return pd.Series([50.0] * len(s), index=s.index)
    return (s - lo) / (hi - lo) * 100


def squad_with_norms(squad, pr_ratings):
    """Add normalised 0-100 columns used by radar, roles, comparison."""
    df = squad.merge(pr_ratings[["player", "PR_score"]], on="player", how="left")
    df["n_press_res"] = _norm(df["PR_score"]).fillna(40)
    df["n_goal"] = _norm(df["goals"])
    df["n_creativity"] = _norm(df["assists"])
    df["n_passing"] = _norm(df["pass_completion_pct"])
    df["n_workrate"] = _norm(df["pressures_applied"])
    return df


def team_radar(xi_with_norms):
    """Average the XI across the 5 axes -> radar values."""
    cols = ["n_press_res", "n_goal", "n_creativity", "n_passing", "n_workrate"]
    return [round(xi_with_norms[c].mean(), 1) for c in cols]


def assign_role(row):
    """Simple rule-based role from the player's normalised profile."""
    g = row.get("pos_group", "MID")
    if g == "GK":
        return "Shot-stopper"
    if g == "DEF":
        return "Ball-playing defender" if row["n_passing"] >= 60 else "Stopper"
    if g == "FWD":
        if row["n_goal"] >= 60: return "Finisher"
        if row["n_creativity"] >= 60: return "Provider"
        return "Outlet"
    # MID
    if row["n_creativity"] >= 65: return "Creator"
    if row["n_workrate"] >= 65: return "Presser / Destroyer"
    if row["n_press_res"] >= 65 and row["n_passing"] >= 60: return "Metronome"
    return "Box-to-box"


def player_strengths(row):
    """Top-2 strengths and bottom-2 weaknesses by normalised score."""
    axes = {"Press Resistance": row["n_press_res"], "Goal Threat": row["n_goal"],
            "Creativity": row["n_creativity"], "Passing": row["n_passing"],
            "Work-rate": row["n_workrate"]}
    ranked = sorted(axes.items(), key=lambda kv: kv[1], reverse=True)
    return [r[0] for r in ranked[:2]], [r[0] for r in ranked[-2:]]


# ════════════════════════════════════════════════════════════════════════════
# PLAYER KPIs, DATA-DERIVED ATTRIBUTES, TRENDS
# ════════════════════════════════════════════════════════════════════════════

DEF_ACTIONS = ["Interception", "Clearance", "Block", "Ball Recovery"]

def player_kpis(events, team):
    """Full per-player KPI table for the squad (descriptive aggregates)."""
    ev = events[events["team"] == team]
    passes = ev[ev["type"] == "Pass"]
    shots = ev[ev["type"] == "Shot"]
    rows = []
    for p in ev["player"].dropna().unique():
        pe = ev[ev["player"] == p]
        pp = passes[passes["player"] == p]
        ps = shots[shots["player"] == p]
        up = pp[pp["under_pressure"] == True]
        rows.append({
            "player": p,
            "apps": pe["match_id"].nunique(),
            "passes": len(pp),
            "pass_pct": round(pp["pass_outcome"].isna().mean()*100, 1) if len(pp) else 0,
            "prog_passes": int((pp["pass_end_location"].notna()).sum()),
            "key_passes": int(pp["pass_shot_assist"].sum()) if "pass_shot_assist" in pp.columns else 0,
            "assists": int(pp["pass_goal_assist"].sum()) if "pass_goal_assist" in pp.columns else 0,
            "shots": len(ps),
            "goals": int((ps["shot_outcome"] == "Goal").sum()),
            "xg": round(float(ps["shot_statsbomb_xg"].sum()), 1) if "shot_statsbomb_xg" in ps.columns else 0,
            "dribbles": len(pe[pe["type"] == "Dribble"]),
            "pressures": len(pe[pe["type"] == "Pressure"]),
            "tackles": len(pe[pe["type"] == "Duel"]),
            "interceptions": len(pe[pe["type"] == "Interception"]),
            "def_actions": len(pe[pe["type"].isin(DEF_ACTIONS)]),
            "pressured_pct": round(up["pass_outcome"].isna().mean()*100, 1) if len(up) else 0,
        })
    return pd.DataFrame(rows).sort_values("apps", ascending=False).reset_index(drop=True)


def player_attributes(kpi_row, ranges):
    """Map KPIs to 6 data-derived attributes (0-99) using squad min-max ranges."""
    def scale(val, key, lo_out=45, hi_out=94):
        lo, hi = ranges[key]
        if hi == lo: return (lo_out+hi_out)//2
        return int(lo_out + (val - lo)/(hi - lo) * (hi_out - lo_out))
    attrs = {
        "SHO": scale(kpi_row["goals"]+kpi_row["xg"], "shooting"),
        "PAS": scale(kpi_row["pass_pct"], "passing"),
        "CRE": scale(kpi_row["assists"]+kpi_row["key_passes"]*0.3, "creativity"),
        "PRE": scale(kpi_row["pressures"], "pressing"),
        "RES": scale(kpi_row["pressured_pct"], "resistance"),
        "DEF": scale(kpi_row["def_actions"], "defending"),
    }
    overall = int(np.mean(sorted(attrs.values(), reverse=True)[:4]))  # top-4 mean
    return attrs, max(min(overall, 94), 52)


def attribute_ranges(kpi_df):
    """Min/max per attribute basis across the squad (for scaling)."""
    return {
        "shooting": (float((kpi_df["goals"]+kpi_df["xg"]).min()), float((kpi_df["goals"]+kpi_df["xg"]).max())),
        "passing": (float(kpi_df["pass_pct"].min()), float(kpi_df["pass_pct"].max())),
        "creativity": (float((kpi_df["assists"]+kpi_df["key_passes"]*0.3).min()), float((kpi_df["assists"]+kpi_df["key_passes"]*0.3).max())),
        "pressing": (float(kpi_df["pressures"].min()), float(kpi_df["pressures"].max())),
        "resistance": (float(kpi_df["pressured_pct"].min()), float(kpi_df["pressured_pct"].max())),
        "defending": (float(kpi_df["def_actions"].min()), float(kpi_df["def_actions"].max())),
    }


def player_trend(events, matches, player, team):
    """Per-match metrics for one player across the season (ordered by match week)."""
    ev = events[(events["team"] == team) & (events["player"] == player)]
    mw = matches[["match_id", "match_week", "opponent"]]
    rows = []
    for mid, me in ev.groupby("match_id"):
        pp = me[me["type"] == "Pass"]
        ps = me[me["type"] == "Shot"]
        rows.append({
            "match_id": mid,
            "passes": len(pp),
            "pass_pct": round(pp["pass_outcome"].isna().mean()*100, 1) if len(pp) else 0,
            "goals": int((ps["shot_outcome"] == "Goal").sum()),
            "shots": len(ps),
            "pressures": len(me[me["type"] == "Pressure"]),
        })
    df = pd.DataFrame(rows).merge(mw, on="match_id", how="left").sort_values("match_week")
    return df


# ════════════════════════════════════════════════════════════════════════════
# TEAM OVERVIEW + AVERAGE POSITIONS
# ════════════════════════════════════════════════════════════════════════════

def team_overview(events, team):
    """Season aggregate team stats."""
    ev = events[events["team"] == team]
    opp = events[events["team"] != team]
    nm = max(ev["match_id"].nunique(), 1)
    passes = ev[ev["type"] == "Pass"]
    total_passes = len(events[events["type"] == "Pass"])
    poss = round(len(passes) / total_passes * 100, 1) if total_passes else 50
    return {
        "matches": nm,
        "possession": poss,
        "passes_pg": round(len(passes)/nm, 0),
        "pass_pct": round(passes["pass_outcome"].isna().mean()*100, 1) if len(passes) else 0,
        "shots_pg": round(len(ev[ev["type"]=="Shot"])/nm, 1),
        "goals_pg": round((ev[ev["type"]=="Shot"]["shot_outcome"]=="Goal").sum()/nm, 1),
        "pressures_pg": round(len(ev[ev["type"]=="Pressure"])/nm, 0),
        "tackles_pg": round(len(ev[ev["type"]=="Duel"])/nm, 0),
    }


def avg_positions(events, team, match_id=None):
    """Average x,y per player (open-play touches) for a formation-style plot."""
    ev = events[(events["team"] == team) & (events["type"].isin(["Pass", "Ball Receipt*", "Carry"]))]
    if match_id is not None:
        ev = ev[ev["match_id"] == match_id]
    ev = ev.dropna(subset=["loc_x", "loc_y"])
    pos = ev.groupby("player").agg(x=("loc_x","mean"), y=("loc_y","mean"),
                                   touches=("loc_x","size")).reset_index()
    return pos.sort_values("touches", ascending=False).head(11)


# ════════════════════════════════════════════════════════════════════════════
# MATCH ANALYSIS: summary + timeline
# ════════════════════════════════════════════════════════════════════════════

def match_summary(events, matches, match_id, team):
    """Headline stats for one match."""
    me = events[events["match_id"] == match_id]
    row = matches[matches["match_id"] == match_id]
    opp = row.iloc[0]["opponent"] if len(row) else "Opponent"
    teams = me["team"].dropna().unique().tolist()
    other = [t for t in teams if t != team]
    other = other[0] if other else "Opponent"

    def side(t):
        s = me[me["team"] == t]
        sh = s[s["type"] == "Shot"]
        pa = s[s["type"] == "Pass"]
        return {
            "shots": len(sh),
            "on_target": int(sh["shot_outcome"].isin(["Goal","Saved"]).sum()) if len(sh) else 0,
            "goals": int((sh["shot_outcome"]=="Goal").sum()),
            "xg": round(float(sh["shot_statsbomb_xg"].sum()), 2) if "shot_statsbomb_xg" in sh.columns else 0,
            "passes": len(pa),
            "pass_pct": round(pa["pass_outcome"].isna().mean()*100, 1) if len(pa) else 0,
            "fouls": len(s[s["type"]=="Foul Committed"]),
        }
    a, b = side(team), side(other)
    total_p = a["passes"] + b["passes"]
    a["poss"] = round(a["passes"]/total_p*100) if total_p else 50
    b["poss"] = 100 - a["poss"]
    return {"team": team, "opponent": other, "home": a, "away": b}


def match_timeline(events, match_id, team):
    """Key events (goals, cards, subs) with minute, ordered."""
    me = events[events["match_id"] == match_id].sort_values(["minute","second"])
    items = []
    for _, r in me.iterrows():
        t = r["type"]
        if t == "Shot" and r.get("shot_outcome") == "Goal":
            items.append((int(r["minute"]), "⚽ Goal", r["team"], r["player"]))
        elif t == "Substitution":
            items.append((int(r["minute"]), "🔁 Sub", r["team"], r["player"]))
        elif t == "Bad Behaviour" and r.get("bad_behaviour_card") in ["Yellow Card","Red Card"]:
            ic = "🟨" if r["bad_behaviour_card"]=="Yellow Card" else "🟥"
            items.append((int(r["minute"]), f"{ic} Card", r["team"], r["player"]))
    return items


# ════════════════════════════════════════════════════════════════════════════
# PRESSING EFFECTIVENESS  (team PPDA + per-player)
# ════════════════════════════════════════════════════════════════════════════

def pressing_effectiveness(events, team):
    """Team PPDA (lower = more aggressive press) + per-player pressing output."""
    ev = events[events["team"] == team]
    opp = events[events["team"] != team]
    nm = max(ev["match_id"].nunique(), 1)

    # PPDA = opponent passes in their build-up zone / our defensive actions there
    opp_passes = opp[(opp["type"] == "Pass") & (opp["loc_x"].notna())]
    # in our pressing zone = opponent's own 60% (their x < 72 in their frame)
    opp_build = opp_passes[opp_passes["loc_x"] < 72]
    our_def = ev[(ev["type"].isin(["Pressure","Duel","Interception","Foul Committed"])) &
                 (ev["loc_x"] > 48)]   # high up our attacking half
    ppda = round(len(opp_build) / max(len(our_def), 1), 1)

    pressures = ev[ev["type"] == "Pressure"]
    high_press = pressures[pressures["loc_x"] > 80]
    rows = []
    for p in pressures["player"].dropna().unique():
        pp = pressures[pressures["player"] == p]
        rows.append({
            "player": p,
            "pressures": len(pp),
            "high_press": len(pp[pp["loc_x"] > 80]),
            "high_press_pct": round(len(pp[pp["loc_x"]>80])/len(pp)*100) if len(pp) else 0,
        })
    per_player = pd.DataFrame(rows).sort_values("pressures", ascending=False).reset_index(drop=True)
    return {
        "ppda": ppda,
        "pressures_pg": round(len(pressures)/nm, 0),
        "high_press_pg": round(len(high_press)/nm, 0),
        "per_player": per_player,
        "pressure_points": pressures.dropna(subset=["loc_x","loc_y"])[["loc_x","loc_y"]],
    }


# ════════════════════════════════════════════════════════════════════════════
# EXPECTED THREAT (xT): data-derived threat surface (descriptive, no ML)
# A simplified xT: each pitch cell's value = goals scored per touch from that
# cell (empirical danger). A pass/carry's xT = value(end) - value(start).
# Anchored to the spatio-temporal lecture (event location + value surface).
# ════════════════════════════════════════════════════════════════════════════

def build_xt_grid(events, nx=12, ny=8):
    """Empirical threat surface: goals-per-touch by pitch cell, normalised 0-1."""
    ev = events.dropna(subset=["loc_x", "loc_y"])
    gx = np.linspace(0, 120, nx + 1)
    gy = np.linspace(0, 80, ny + 1)
    cx = np.clip(np.digitize(ev["loc_x"], gx) - 1, 0, nx - 1)
    cy = np.clip(np.digitize(ev["loc_y"], gy) - 1, 0, ny - 1)
    touches = np.zeros((ny, nx))
    for i, j in zip(cy, cx):
        touches[i, j] += 1
    goals = ev[(ev["type"] == "Shot") & (ev["shot_outcome"] == "Goal")]
    gxx = np.clip(np.digitize(goals["loc_x"], gx) - 1, 0, nx - 1)
    gyy = np.clip(np.digitize(goals["loc_y"], gy) - 1, 0, ny - 1)
    gcount = np.zeros((ny, nx))
    for i, j in zip(gyy, gxx):
        gcount[i, j] += 1
    grid = np.divide(gcount, touches, out=np.zeros_like(gcount), where=touches > 5)
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid, gx, gy


def _cell(x, y, gx, gy):
    nx, ny = len(gx) - 1, len(gy) - 1
    return (int(np.clip(np.digitize([x], gx)[0] - 1, 0, nx - 1)),
            int(np.clip(np.digitize([y], gy)[0] - 1, 0, ny - 1)))


def xt_added(events, team, grid, gx, gy):
    """Rank players by total Expected Threat added through passes & carries."""
    moves = events[(events["team"] == team) & (events["type"].isin(["Pass", "Carry"]))]
    rows = {}
    for _, r in moves.iterrows():
        if r["type"] == "Pass":
            if pd.notna(r.get("pass_outcome")):
                continue
            end = r.get("pass_end_location")
        else:
            end = r.get("carry_end_location")
        start = r.get("location")
        if not (isinstance(end, (list, np.ndarray)) and isinstance(start, (list, np.ndarray))):
            continue
        sx, sy = _cell(start[0], start[1], gx, gy)
        ex, ey = _cell(end[0], end[1], gx, gy)
        delta = grid[ey, ex] - grid[sy, sx]
        p = r["player"]
        rows.setdefault(p, 0.0)
        if delta > 0:
            rows[p] += delta
    df = pd.DataFrame([{"player": k, "xt_added": round(v, 2)} for k, v in rows.items()])
    if len(df) == 0:
        return df
    return df.sort_values("xt_added", ascending=False).reset_index(drop=True)


def player_xt_zone_grid(events, team, player, grid, gx, gy):
    """Per-zone breakdown of one player's Expected Threat added this season,
    attributed to the zone each pass/carry started in. Same grid shape as
    build_xt_grid, so it can be drawn with the same pitch overlay."""
    nx, ny = len(gx) - 1, len(gy) - 1
    out = np.zeros((ny, nx))
    moves = events[(events["team"] == team) & (events["player"] == player) &
                   (events["type"].isin(["Pass", "Carry"]))]
    for _, r in moves.iterrows():
        if r["type"] == "Pass":
            if pd.notna(r.get("pass_outcome")):
                continue
            end = r.get("pass_end_location")
        else:
            end = r.get("carry_end_location")
        start = r.get("location")
        if not (isinstance(end, (list, np.ndarray)) and isinstance(start, (list, np.ndarray))):
            continue
        sx, sy = _cell(start[0], start[1], gx, gy)
        ex, ey = _cell(end[0], end[1], gx, gy)
        delta = grid[ey, ex] - grid[sy, sx]
        if delta > 0:
            out[sy, sx] += delta
    return out


# ════════════════════════════════════════════════════════════════════════════
# MODEL VALIDATION: backtest Elo vs actual results (Stefani benchmark)
# ════════════════════════════════════════════════════════════════════════════

def backtest_elo(all_matches, k=20, initial=1000, home_adv=40):
    """Walk the season chronologically: predict with Elo BEFORE each game, then update.
    Returns accuracy on decisive games vs a home-win baseline."""
    teams = sorted(set(all_matches["home_team"]) | set(all_matches["away_team"]))
    elo = {t: initial for t in teams}
    correct = total = home_correct = 0
    for _, m in all_matches.sort_values("match_id").iterrows():
        h, a = m["home_team"], m["away_team"]
        hs, as_ = m["home_score"], m["away_score"]
        if hs == as_:
            # update only, skip from accuracy (draws not predicted)
            pass
        else:
            pred_home = (elo[h] + home_adv) >= elo[a]
            home_won = hs > as_
            if pred_home == home_won:
                correct += 1
            if home_won:
                home_correct += 1
            total += 1
        # Elo update
        dr = (elo[h] + home_adv) - elo[a]
        we = 1 / (10 ** (-dr / 400) + 1)
        w = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        elo[h] += k * (w - we)
        elo[a] += k * ((1 - w) - (1 - we))
    return {
        "accuracy": round(correct / total * 100, 1) if total else 0,
        "home_baseline": round(home_correct / total * 100, 1) if total else 0,
        "games": total, "benchmark": 67,
    }


def role_familiarity(row):
    """+/++ familiarity tag from the player's strongest attribute."""
    top = max(row.get("n_press_res",0), row.get("n_goal",0), row.get("n_creativity",0),
              row.get("n_passing",0), row.get("n_workrate",0))
    return "++" if top >= 75 else ("+" if top >= 58 else "")


TACTICAL_PRESETS = {
    "Auto (vs opponent)": None,
    "Tiki-Taka":   {"press_resistance":0.30,"goal_threat":0.20,"passing":0.40,"pressing":0.10},
    "Gegenpress":  {"press_resistance":0.30,"goal_threat":0.20,"passing":0.10,"pressing":0.40},
    "Counter":     {"press_resistance":0.20,"goal_threat":0.50,"passing":0.15,"pressing":0.15},
    "Balanced":    {"press_resistance":0.25,"goal_threat":0.25,"passing":0.25,"pressing":0.25},
}


# ════════════════════════════════════════════════════════════════════════════
# PRESEASON SCOUTING GRADES (Manchester City): the starting point each
# player's data-derived overall is adjusted from.
# ════════════════════════════════════════════════════════════════════════════

BASELINE_RATINGS = {
    "Agüero": 87, "David": 88, "Silva": 88, "De Bruyne": 86, "Touré": 85,
    "Kompany": 85, "Otamendi": 84, "Hart": 83, "Nasri": 83, "Zabaleta": 82,
    "Sterling": 82, "Demichelis": 81, "Bony": 81, "Navas": 81, "Mangala": 80,
    "Clichy": 80, "Fernandinho": 80, "Sagna": 79, "Fernando": 79, "Kolarov": 79,
    "Caballero": 78, "Delph": 77, "Zuculini": 74, "Denayer": 71, "Iheanacho": 67,
    "Wright": 63, "Roberts": 65, "Lopes": 73,
}

def baseline_overall(name):
    """Find the preseason scouting grade for a StatsBomb player name (by surname/keyword)."""
    n = str(name)
    # special cases first
    if "Fernando Luiz" in n or "Roza" in n or "Rosa" in n:
        return BASELINE_RATINGS["Fernandinho"]
    if "Caballero" in n: return BASELINE_RATINGS["Caballero"]
    for key, rating in BASELINE_RATINGS.items():
        if key in n:
            return rating
    return None  # unknown -> caller falls back to data-derived


# ════════════════════════════════════════════════════════════════════════════
# FORMATION SLOTS: proper named positions on a vertical pitch (x:0-80, y:0-120)
# Fixes cluttered/!= placement. Each slot has a pitch coord + position group.
# ════════════════════════════════════════════════════════════════════════════

FORMATION_SLOTS = {
    "4-3-3": [
        ("GK", 40, 8, "GK"),
        ("LB", 12, 30, "DEF"), ("LCB", 30, 24, "DEF"), ("RCB", 50, 24, "DEF"), ("RB", 68, 30, "DEF"),
        ("LCM", 24, 58, "MID"), ("CM", 40, 50, "MID"), ("RCM", 56, 58, "MID"),
        ("LW", 14, 95, "FWD"), ("ST", 40, 100, "FWD"), ("RW", 66, 95, "FWD"),
    ],
    "4-4-2": [
        ("GK", 40, 8, "GK"),
        ("LB", 12, 30, "DEF"), ("LCB", 30, 24, "DEF"), ("RCB", 50, 24, "DEF"), ("RB", 68, 30, "DEF"),
        ("LM", 12, 62, "MID"), ("LCM", 30, 56, "MID"), ("RCM", 50, 56, "MID"), ("RM", 68, 62, "MID"),
        ("LST", 30, 98, "FWD"), ("RST", 50, 98, "FWD"),
    ],
    "3-5-2": [
        ("GK", 40, 8, "GK"),
        ("LCB", 24, 24, "DEF"), ("CB", 40, 20, "DEF"), ("RCB", 56, 24, "DEF"),
        ("LWB", 8, 60, "MID"), ("LCM", 26, 54, "MID"), ("CM", 40, 48, "MID"), ("RCM", 54, 54, "MID"), ("RWB", 72, 60, "MID"),
        ("LST", 30, 98, "FWD"), ("RST", 50, 98, "FWD"),
    ],
}

def formation_slots(formation_name):
    return FORMATION_SLOTS.get(formation_name, FORMATION_SLOTS["4-3-3"])


def fill_slots(formation_name, chosen_by_group):
    """Assign chosen players to the formation's slots, in order, per group.
    chosen_by_group: {'GK':[...], 'DEF':[...], 'MID':[...], 'FWD':[...]}.
    Returns list of dicts: slot,label,x,y,group,player."""
    slots = formation_slots(formation_name)
    idx = {"GK":0,"DEF":0,"MID":0,"FWD":0}
    out = []
    for label, x, y, grp in slots:
        players = chosen_by_group.get(grp, [])
        player = players[idx[grp]] if idx[grp] < len(players) else None
        idx[grp] += 1
        out.append({"label":label, "x":x, "y":y, "group":grp, "player":player})
    return out


# ════════════════════════════════════════════════════════════════════════════
# WHO TO STOP (fixed): exclude GKs, prefer midfielders for the headline
# ════════════════════════════════════════════════════════════════════════════

def _opp_positions(events, opponent):
    """Map opponent player -> position from their Starting XI lineups."""
    sxi = events[(events["type"] == "Starting XI") & (events["team"] == opponent)]
    pos = {}
    for _, r in sxi.iterrows():
        t = r["tactics"]
        if isinstance(t, dict):
            for pl in t.get("lineup", []):
                nm = pl.get("player", {}).get("name")
                ps = pl.get("position", {}).get("name", "")
                if nm and nm not in pos:
                    pos[nm] = ps
    return pos


def who_to_stop(events, opponent, centrality_df):
    """Filter centrality to outfield players, prefer a midfielder as the hub."""
    pos = _opp_positions(events, opponent)
    df = centrality_df.copy()
    df["position"] = df["player"].map(pos).fillna("")
    # drop goalkeepers entirely
    df = df[~df["position"].str.contains("Goalkeeper", na=False)]
    df = df.reset_index(drop=True)
    # headline hub: highest centrality among midfielders/wingers if any, else top
    mids = df[df["position"].str.contains("Midfield|Wing|Center Forward", na=False)]
    hub = (mids.iloc[0] if len(mids) else (df.iloc[0] if len(df) else None))
    return df, hub


# ════════════════════════════════════════════════════════════════════════════
# SEASON PREDICTOR: Monte Carlo simulation (real prediction)
# Uses Elo to drive a Poisson goal model; simulates a full season many times.
# ════════════════════════════════════════════════════════════════════════════

def monte_carlo_season(elo_df, our_team="Manchester City", n_sims=400,
                       home_adv=60, base_home=1.45, base_away=1.15, seed=42):
    """Simulate a full double round-robin season N times from Elo. Returns
    predicted finish distribution + key probabilities for our team."""
    rng = np.random.default_rng(seed)
    teams = elo_df["team"].tolist()
    elo = dict(zip(elo_df["team"], elo_df["elo"]))
    n = len(teams)
    idx = {t: i for i, t in enumerate(teams)}

    # Pre-compute expected goals for every ordered fixture (home, away)
    fixtures = [(h, a) for h in teams for a in teams if h != a]
    lam_h = np.zeros(len(fixtures)); lam_a = np.zeros(len(fixtures))
    fi_h = np.zeros(len(fixtures), dtype=int); fi_a = np.zeros(len(fixtures), dtype=int)
    for k, (h, a) in enumerate(fixtures):
        dr = (elo[h] + home_adv) - elo[a]
        lam_h[k] = np.clip(base_home + dr / 280, 0.2, 4.5)
        lam_a[k] = np.clip(base_away - dr / 280, 0.2, 4.5)
        fi_h[k] = idx[h]; fi_a[k] = idx[a]

    finishes = []; points_list = []
    for _ in range(n_sims):
        gh = rng.poisson(lam_h); ga = rng.poisson(lam_a)
        pts = np.zeros(n)
        # award points
        home_win = gh > ga; away_win = ga > gh; draw = gh == ga
        np.add.at(pts, fi_h[home_win], 3)
        np.add.at(pts, fi_a[away_win], 3)
        np.add.at(pts, fi_h[draw], 1)
        np.add.at(pts, fi_a[draw], 1)
        # rank (1 = most points)
        order = np.argsort(-pts)
        rank = {teams[order[r]]: r + 1 for r in range(n)}
        finishes.append(rank[our_team])
        points_list.append(pts[idx[our_team]])

    finishes = np.array(finishes); points_list = np.array(points_list)
    # distribution of finishing positions
    dist = pd.Series(finishes).value_counts(normalize=True).sort_index()
    return {
        "avg_finish": round(float(finishes.mean()), 1),
        "avg_points": round(float(points_list.mean()), 1),
        "p_title": round(float((finishes == 1).mean()) * 100, 1),
        "p_top4": round(float((finishes <= 4).mean()) * 100, 1),
        "p_top1_to_3": round(float((finishes <= 3).mean()) * 100, 1),
        "p_releg": round(float((finishes >= 18).mean()) * 100, 1),
        "best": int(finishes.min()), "worst": int(finishes.max()),
        "dist": dist, "n_sims": n_sims,
    }


# ════════════════════════════════════════════════════════════════════════════
# LIVE MATCH ASSISTANT: in-game prescriptive suggestions
# ════════════════════════════════════════════════════════════════════════════

def match_assistant(profile, key_player, fatigue, danger, minute, game_state):
    """Return prioritised in-game suggestions from scouted data + game state.
    game_state in {'Leading','Level','Trailing'}."""
    tips = []
    late = minute >= 70
    kp = key_player["player"] if key_player else "their playmaker"

    # always-on: contain the danger man
    if danger is not None and len(danger):
        tips.append(("info", "Containment",
            f"Keep a man on {danger.iloc[0]['player']}, their biggest threat all game."))

    if game_state == "Trailing":
        if fatigue.get("fades"):
            tips.append(("good", "Go for it",
                f"They drop ~{abs(fatigue['shift']):.0f} yds deeper after 75'. Push your full-backs high "
                f"and target the space in behind, bring on a pacey forward."))
        else:
            tips.append(("good", "Increase risk",
                "Add an attacker, overload central areas, and quicken your tempo to break them down."))
        tips.append(("info", "Set pieces",
            "Commit numbers to attacking set pieces, a third of late goals come from dead balls."))
    elif game_state == "Leading":
        if late:
            tips.append(("warn", "Game-manage",
                f"Protect the lead: drop your line a touch, keep {kp} marked, and use subs to kill tempo."))
        else:
            tips.append(("info", "Stay disciplined",
                f"Hold your shape and keep cutting the supply to {kp}; don't over-commit."))
    else:  # Level
        tips.append(("info", "Find the edge",
            f"It's tight, exploit their weaker flank and keep pressing {kp} off the ball."))
        if late and fatigue.get("fades"):
            tips.append(("good", "Late push",
                "They're tiring, a fresh runner now could win it in the final 15."))

    # press reminder
    if profile.get("press_level") == "High":
        tips.append(("warn", "Beat the press",
            "They press high, play through the first line quickly rather than going long."))
    return tips


# ════════════════════════════════════════════════════════════════════════════
# PRESCRIPTIVE SEASON METRICS: series + projection per chosen metric
# ════════════════════════════════════════════════════════════════════════════

def season_metric_series(events, matches, metric):
    """Return per-matchweek series for the chosen metric + a projected end value."""
    cm = matches.sort_values("match_week").copy()
    cm["pts"] = cm["result"].map({"W":3,"D":1,"L":0})
    ev = events
    rows = []
    for _, m in cm.iterrows():
        mid = m["match_id"]; me = ev[(ev["match_id"] == mid)]
        city = me[me["team"] == "Manchester City"]
        passes = city[city["type"] == "Pass"]
        val = {
            "Points": m["pts"],
            "Goals For": m["city_score"],
            "Goals Against": m["opp_score"],
            "Possession %": round(len(passes) / max(len(me[me["type"]=="Pass"]),1) * 100, 1),
            "Passes": len(passes),
            "Shots": len(city[city["type"] == "Shot"]),
        }.get(metric, 0)
        rows.append({"match_week": m["match_week"], "opponent": m["opponent"], "value": val,
                     "result": m["result"]})
    df = pd.DataFrame(rows)
    # momentum = rolling 5-game average of points
    if metric == "Momentum":
        pts = cm["pts"].reset_index(drop=True)
        df["value"] = pts.rolling(5, min_periods=1).mean().round(2).values
    df["rolling"] = df["value"].rolling(5, min_periods=1).mean().round(2)
    recent = df["value"].tail(5).mean()
    if metric == "Points":
        projected_total = round(cm["pts"].sum() / len(cm) * 38, 0)
    else:
        projected_total = round(recent, 2)
    return df, projected_total


# ════════════════════════════════════════════════════════════════════════════
# FORM DIP DETECTION: find the worst rolling-window patch of the season and
# describe what was actually happening in it (goals for/against, fixture mix).
# ════════════════════════════════════════════════════════════════════════════

def detect_dip(matches, window=5):
    """Locate the worst `window`-game rolling-form patch this season."""
    cm = matches.sort_values("match_week").copy()
    cm["pts"] = cm["result"].map({"W": 3, "D": 1, "L": 0})
    cm["pts_roll"] = cm["pts"].rolling(window, min_periods=window).mean()
    if cm["pts_roll"].notna().sum() == 0:
        return None
    worst_idx = cm["pts_roll"].idxmin()
    end_week = int(cm.loc[worst_idx, "match_week"])
    win = cm[(cm["match_week"] > end_week - window) & (cm["match_week"] <= end_week)]
    gf, ga = int(win["city_score"].sum()), int(win["opp_score"].sum())
    home_n = int((win["home_or_away"] == "Home").sum())
    return {
        "start_week": int(win["match_week"].min()), "end_week": end_week,
        "season_avg_pts": round(float(cm["pts"].mean()), 2),
        "dip_avg_pts": round(float(win["pts"].mean()), 2),
        "gf": gf, "ga": ga, "opponents": win["opponent"].tolist(),
        "results": win["result"].tolist(),
        "home_n": home_n, "away_n": len(win) - home_n,
    }


# ════════════════════════════════════════════════════════════════════════════
# SQUAD AVAILABILITY & LOAD
# StatsBomb has NO medical/injury database. We build this honestly from the real
# availability signals it DOES log: injury substitutions, players-off-for-
# treatment, cards/suspensions, plus a clearly-labelled minutes/load proxy.
# ════════════════════════════════════════════════════════════════════════════

def _lineup_names(tactics):
    if isinstance(tactics, dict):
        return [pl.get("player", {}).get("name") for pl in tactics.get("lineup", [])]
    return []

def squad_availability(events, matches, team):
    """Per-player availability + workload from real injury/discipline events."""
    ev = events[events["team"] == team]
    last5 = set(matches.sort_values("match_week").tail(5)["match_id"])

    minutes, apps, recent, injury = {}, {}, {}, {}
    for mid, me in ev.groupby("match_id"):
        starters = []
        for _, r in me[me["type"] == "Starting XI"].iterrows():
            starters = _lineup_names(r["tactics"])
        end = min(int(me["minute"].max()) + 1, 98) if len(me) else 95
        offs, ons = {}, {}
        for _, r in me[me["type"] == "Substitution"].iterrows():
            offs[r["player"]] = int(r["minute"])
            rep = r.get("substitution_replacement")
            nm = rep.get("name") if isinstance(rep, dict) else (rep if isinstance(rep, str) else None)
            if nm: ons[nm] = int(r["minute"])
            if r.get("substitution_outcome") == "Injury":
                injury[r["player"]] = injury.get(r["player"], 0) + 1
        for _, r in me[me["type"] == "Player Off"].iterrows():
            offs.setdefault(r["player"], int(r["minute"]))
        for p in set(starters) | set(ons.keys()):
            if not p: continue
            start = 0 if p in starters else ons.get(p, end)
            mins = max(0, offs.get(p, end) - start)
            minutes[p] = minutes.get(p, 0) + mins
            apps[p] = apps.get(p, 0) + 1
            if mid in last5: recent[p] = recent.get(p, 0) + mins

    yellows, reds = {}, {}
    for _, r in ev[ev["type"] == "Bad Behaviour"].iterrows():
        c = r.get("bad_behaviour_card")
        if c == "Yellow Card": yellows[r["player"]] = yellows.get(r["player"], 0) + 1
        elif c in ("Red Card", "Second Yellow"): reds[r["player"]] = reds.get(r["player"], 0) + 1

    rows = []
    for p in minutes:
        mins, ap, rec = minutes[p], apps.get(p, 0), recent.get(p, 0)
        inj, y, rd = injury.get(p, 0), yellows.get(p, 0), reds.get(p, 0)
        load = min(100, int(rec / 450 * 60 + mins / 3200 * 40))   # 0-100 workload proxy
        if rd > 0: status = "🔴 Suspended"
        elif y >= 4: status = "🟡 Booking risk"
        elif inj > 0: status = "🩹 Knock"
        elif load >= 75: status = "🔵 High load"
        else: status = "🟢 Available"
        rows.append({"player": p, "apps": ap, "minutes": mins, "recent5_min": rec,
                     "injury_subs": inj, "yellows": y, "reds": rd, "load": load, "status": status})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("minutes", ascending=False).reset_index(drop=True)
    return df


# ════════════════════════════════════════════════════════════════════════════
# DERIVED OVERALL RATING (a preseason baseline, adjusted by real data)
# overall = α · performance(StatsBomb) + (1-α) · preseason baseline
#   performance = position-weighted blend of the 6 data attributes
#   α           = 0.5 · confidence,  confidence = min(1, apps/19)
# So well-sampled players move toward their actual output; low-minute players
# stay near the baseline. Numbers stay sensible but are earned from the data.
# ════════════════════════════════════════════════════════════════════════════

POS_WEIGHTS = {
    "GK":  {"DEF":0.22, "RES":0.24, "PAS":0.20, "PRE":0.10, "CRE":0.10, "SHO":0.14},
    "DEF": {"DEF":0.30, "RES":0.22, "PAS":0.20, "PRE":0.13, "CRE":0.10, "SHO":0.05},
    "MID": {"PAS":0.25, "CRE":0.24, "RES":0.20, "PRE":0.16, "DEF":0.10, "SHO":0.05},
    "FWD": {"SHO":0.34, "CRE":0.24, "PAS":0.16, "RES":0.10, "PRE":0.06, "DEF":0.10},
}

def squad_overalls(kpis, ranges, positions, scale=0.32, max_shift=5):
    """Derive each player's overall from the preseason baseline + a data adjustment.

      performance = Σ position_weight · attribute        (position-weighted form)
      delta       = performance − positional average     (better/worse than peers)
      adjustment  = scale · delta · confidence           (capped at ±max_shift)
      overall     = preseason baseline + adjustment

    confidence = min(1, apps/19) so low-minute players stay near the baseline;
    goalkeepers are anchored to the baseline (outfield attributes can't rate them).
    Returns {player: {overall, base, perf, delta, adj, conf, pos}}.
    """
    posmap = dict(zip(positions["player"], positions["pos_group"]))
    recs = []
    for _, row in kpis.iterrows():
        n = row["player"]; attrs, _ = player_attributes(row, ranges)
        g = posmap.get(n, "MID"); w = POS_WEIGHTS.get(g, POS_WEIGHTS["MID"])
        perf = sum(w.get(k, 0) * v for k, v in attrs.items())
        recs.append({"player": n, "pos": g, "base": baseline_overall(n),
                     "perf": perf, "apps": int(row["apps"])})
    df = pd.DataFrame(recs)
    df["delta"] = df["perf"] - df.groupby("pos")["perf"].transform("mean")
    df["conf"] = (df["apps"] / 19.0).clip(upper=1.0)

    out = {}
    for _, r in df.iterrows():
        base = r["base"]
        if pd.isna(base):                      # youth / unknown -> positional baseline
            grp = df[(df["pos"] == r["pos"])]["base"].dropna()
            base = float(grp.mean()) if len(grp) else 72.0
        if r["pos"] == "GK":
            adj = 0.0                          # anchor keepers to baseline
        else:
            adj = max(min(scale * r["delta"] * r["conf"], max_shift), -max_shift)
        overall = int(round(max(min(base + adj, 94), 50)))
        out[r["player"]] = {"overall": overall, "base": (None if pd.isna(r["base"]) else int(r["base"])),
                            "perf": round(r["perf"], 1), "delta": round(r["delta"], 1),
                            "adj": round(adj, 1), "conf": round(r["conf"], 2), "pos": r["pos"]}
    return out
