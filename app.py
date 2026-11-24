"""
PitchInstinct, Manchester City Manager's Decision System
EA Sports-inspired presentation. Descriptive analytics only (no ML).
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from mplsoccer import Pitch, VerticalPitch
import plotly.graph_objects as go
import plotly.express as px
import sys, os, threading, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import (get_all_events, get_city_matches, get_city_squad_stats,
get_player_positions, get_opponent_events, get_team_full_matches,
get_opponents, get_all_pl_matches, prefetch_team_full_matches, OUR_TEAM)
import news as N
from analytics import (pressure_resistance_rating, pressure_resistance_zones,
tactical_degradation, transition_vulnerability,
passing_network_centrality, passing_network_edges,
set_piece_analysis, compute_elo_ratings, compute_best_xi_score)
from scout import (scout_opponent, scout_key_player, scout_fatigue,
battlegrounds, opponent_priorities, recommend_xi)
import features as F
import predictive as P
import spatial as SP
import spatial_cv as CV
import ui
import player_images as pimg
from glossary import GLOSSARY, PAGE_TERMS

st.set_page_config(page_title="PitchInstinct", page_icon="⚽", layout="wide")
ui.inject_theme()

# stadium photo backdrop behind the frosted glass (if present in the folder)
import base64
for _bgn in ("background.jpg", "background.png"):
    _bgp = os.path.join(os.path.dirname(__file__), _bgn)
    if os.path.exists(_bgp):
        with open(_bgp, "rb") as _bf:
            ui.app_background(base64.b64encode(_bf.read()).decode())
        break

# manager portrait (Pep) for the dashboard hero
PEP_IMG = None
_pep = os.path.join(os.path.dirname(__file__), "pep.jpg")
if os.path.exists(_pep):
    with open(_pep, "rb") as _pf:
        PEP_IMG = base64.b64encode(_pf.read()).decode()

BADGE = "https://upload.wikimedia.org/wikipedia/en/e/eb/Manchester_City_FC_badge.svg"
FORMATIONS = {"4-3-3": {"GK":1,"DEF":4,"MID":3,"FWD":3},
"4-4-2": {"GK":1,"DEF":4,"MID":4,"FWD":2},
"3-5-2": {"GK":1,"DEF":3,"MID":5,"FWD":2}}

POS_SHORT = {"Goalkeeper":"GK","Right Back":"RB","Left Back":"LB","Right Center Back":"CB",
"Left Center Back":"CB","Center Back":"CB","Right Defensive Midfield":"CDM",
"Left Defensive Midfield":"CDM","Center Defensive Midfield":"CDM",
"Right Center Midfield":"CM","Left Center Midfield":"CM","Center Midfield":"CM",
"Right Midfield":"RM","Left Midfield":"LM","Center Attacking Midfield":"CAM",
"Right Wing":"RW","Left Wing":"LW","Right Center Forward":"ST",
"Left Center Forward":"ST","Center Forward":"ST","Secondary Striker":"CF"}


# ── data (cached) ─────────────────────────────────────────────────────────────
# Lite build: the raw event frames are shared (cache_resource) rather than copied on
# every rerun, which keeps memory flat on a small hosted server. Nothing mutates them.
@st.cache_resource(show_spinner=False)
def load_city():
    ev = get_all_events(); mt = get_city_matches(); sq = get_city_squad_stats()
    pos = get_player_positions(); opps = get_opponents()
    return ev, mt, sq, pos, opps

@st.cache_data(show_spinner=False)
def derived(_ev, _sq):
    pr = pressure_resistance_rating(_ev, OUR_TEAM)
    kpis = F.player_kpis(_ev, OUR_TEAM)
    ranges = F.attribute_ranges(kpis)
    sn = F.squad_with_norms(_sq, pr)
    return pr, kpis, ranges, sn

with st.spinner("Loading Manchester City..."):
    city_events, city_matches, squad, positions, opponents = load_city()
    pr_ratings, kpis, attr_ranges, squad_norm = derived(city_events, squad)
squad_norm = squad_norm.merge(positions[["player","pos_group","position"]], on="player", how="left")
squad_norm["pos_group"] = squad_norm["pos_group"].fillna("MID")
OVERALLS = F.squad_overalls(kpis, attr_ranges, positions) # derived, baseline-anchored ratings

@st.cache_data(show_spinner=False)
def load_availability(_ev, _mt):
    return F.squad_availability(_ev, _mt, OUR_TEAM)
availability = load_availability(city_events, city_matches)

@st.cache_data(show_spinner=False)
def load_all_pl_matches():
    return get_all_pl_matches()
all_pl_matches = load_all_pl_matches()

@st.cache_resource(show_spinner=False)
def load_opp_full(opp): return get_team_full_matches(opp)
@st.cache_resource(show_spinner=False)
def load_opp_vs(opp): return get_opponent_events(opp)
@st.cache_data(show_spinner=False)
def load_elo(): return compute_elo_ratings(get_all_pl_matches())

# ── per-opponent scouting (cached by opponent name) ───────────────────────────
# These were previously recomputed on every rerun (every slider/tab/button), which
# was the main source of the lag when navigating. Keyed by the opponent string; the
# heavy event frame is pulled from the already-cached load_opp_full, so cache keys
# stay cheap. @st.cache_data hands back a copy each call, so downstream code is free
# to read/mutate the results without corrupting the cache.
@st.cache_data(show_spinner=False)
def load_scout(opp):
    ev = load_opp_full(opp)
    return scout_opponent(ev, opp), scout_key_player(ev, opp), scout_fatigue(ev, opp)
@st.cache_data(show_spinner=False)
def load_danger(opp):    return F.danger_men(load_opp_full(opp), opp)
@st.cache_data(show_spinner=False)
def load_setpiece(opp):  return F.set_piece_profile(load_opp_full(opp), opp)
@st.cache_data(show_spinner=False)
def load_gamestate(opp): return F.game_state_tendencies(load_opp_full(opp), opp)
@st.cache_data(show_spinner=False)
def load_scout_report(opp):
    ev = load_opp_full(opp)
    return {"danger": load_danger(opp), "likely_xi": F.likely_xi(ev, opp),
            "set_piece": load_setpiece(opp), "pressing": F.pressing_triggers(ev, opp),
            "game_state": load_gamestate(opp)}
@st.cache_data(show_spinner=False)
def load_who_to_stop(opp):
    ev = load_opp_full(opp)
    cent = passing_network_centrality(ev, opp)
    edges = passing_network_edges(ev, opp)
    ranked, hub = F.who_to_stop(ev, opp, cent)
    return cent, edges, ranked, hub
@st.cache_data(show_spinner=False)
def load_transition(opp, win):
    return transition_vulnerability(load_opp_vs(opp), opp, window_sec=win)
@st.cache_data(show_spinner=False)
def load_degradation(opp):
    return tactical_degradation(load_opp_full(opp), opp)

# ── background prefetch ───────────────────────────────────────────────────────
# Quietly pull every opponent's full-season data to disk in a daemon thread so a
# later visit never has to wait on a StatsBomb fetch. Teams already cached are
# skipped instantly. @st.cache_resource makes this run once per server process
# (not per rerun). The thread only touches data_loader (no Streamlit calls).
def _prefetch_opponents(opp_list):
    for o in opp_list:
        try:
            prefetch_team_full_matches(o)
        except Exception:
            pass

@st.cache_resource(show_spinner=False)
def _start_prefetch(opp_key):
    th = threading.Thread(target=_prefetch_opponents, args=(list(opp_key),),
                          name="pi-prefetch", daemon=True)
    th.start()
    return th
_start_prefetch(tuple(opponents))   # warms every opponent's data to disk in the background

# ── trained ML models (train once, cache) ────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_xg_model(_ev):
    return P.train_xg_model(_ev)
@st.cache_resource(show_spinner=False)
def load_match_model(_ev, _mt):
    return P.train_match_model(_ev, _mt)
@st.cache_resource(show_spinner=False)
def load_form_model(_ev, _mt):
    return P.train_form_model(_ev, _mt)

def pos_short(p):
    return POS_SHORT.get(str(p), "SUB")

SHORT_NAMES = {
"Sergio Leonel Agüero del Castillo":"Agüero", "David Josué Jiménez Silva":"Silva",
"Gnégnéri Yaya Touré":"Touré", "Nicolás Hernán Otamendi":"Otamendi",
"Jesús Navas González":"Navas", "Fernando Luiz Rosa":"Fernandinho",
"Wilfredo Daniel Caballero":"Caballero", "Aleksandar Kolarov":"Kolarov",
"Bacary Sagna":"Sagna", "Fabian Delph":"Delph", "Raheem Sterling":"Sterling",
"Kevin De Bruyne":"De Bruyne", "Martín Demichelis":"Demichelis",
"Wilfried Bony":"Bony", "Kelechi Iheanacho":"Iheanacho", "Samir Nasri":"Nasri",
"Eliaquim Mangala":"Mangala", "Vincent Kompany":"Kompany", "Joe Hart":"Hart",
"Pablo Zabaleta":"Zabaleta", "Gaël Clichy":"Clichy",
}
def short_name(name):
    if name in SHORT_NAMES:
        return SHORT_NAMES[name]
    return str(name).split()[-1]

def player_card_html(name):
    krow = kpis[kpis["player"] == name]
    prow = positions[positions["player"] == name]
    if len(krow) == 0:
        return ""
    attrs, data_ovr = F.player_attributes(krow.iloc[0], attr_ranges)
    ovr = OVERALLS.get(name, {}).get("overall", data_ovr) # derived, baseline-anchored rating
    posn = pos_short(prow.iloc[0]["position"]) if len(prow) else "SUB"
    pgrp = prow.iloc[0]["pos_group"] if len(prow) and "pos_group" in prow.columns else "MID"
    img = pimg.player_image(name, pgrp, 256)
    return ui.fut_card(short_name(name), ovr, posn, attrs, image=img)

def player_overall(name):
    if name in OVERALLS:
        return OVERALLS[name]["overall"]
    krow = kpis[kpis["player"] == name]
    if len(krow) == 0: return 50
    _, ovr = F.player_attributes(krow.iloc[0], attr_ranges)
    return ovr


    # ── plot helpers ──────────────────────────────────────────────────────────────
_METRIC_EVENT_TYPE = {"goals": "Shot", "shots": "Shot", "pressures": "Pressure",
"passes": "Pass", "pass_pct": "Pass"}

def player_metric_points(events, player, metric):
    """Locations behind one player's chosen metric, for the Player Profile pitch map.
    Returns (points_df, mode) where mode is 'success' (pass_pct, colour by outcome)
    or 'kde' (a density map for everything else)."""
    pe = events[events["player"] == player]
    etype = _METRIC_EVENT_TYPE.get(metric, "Pass")
    d = pe[pe["type"] == etype].dropna(subset=["loc_x", "loc_y"])
    if metric == "goals":
        d = d[d["shot_outcome"] == "Goal"]
    if metric == "pass_pct":
        d = d.copy(); d["success"] = d["pass_outcome"].isna()
        return d[["loc_x", "loc_y", "success"]], "success"
    return d[["loc_x", "loc_y"]], "kde"


def pitch_kde(points, xcol, ycol, cmap, title=""):
    fig, ax = plt.subplots(figsize=(7.5,5)); fig.patch.set_facecolor("#0E1B33")
    p = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); p.draw(ax=ax)
    if len(points): p.kdeplot(points[xcol], points[ycol], ax=ax, cmap=cmap, levels=10, fill=True, alpha=.6)
    if title: ax.set_title(title, color="#E8EEF6", fontsize=9)
    return fig


# Cached version of the KDE pitch map. The gaussian-KDE render is ~0.7s and the same
# maps get redrawn on every rerun; here we render once and cache the PNG, keyed by the
# raw point coordinates (passed as bytes, which Streamlit hashes cheaply). Identical
# data -> instant image instead of re-running matplotlib. Visually identical to
# pitch_kde (same figure size/colours/levels), just served as a picture.
from io import BytesIO

@st.cache_data(show_spinner=False, max_entries=256)
def _pitch_kde_png(xb, yb, cmap, title):
    xs = np.frombuffer(xb, dtype="float64"); ys = np.frombuffer(yb, dtype="float64")
    fig, ax = plt.subplots(figsize=(7.5,5)); fig.patch.set_facecolor("#0E1B33")
    p = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); p.draw(ax=ax)
    if len(xs): p.kdeplot(xs, ys, ax=ax, cmap=cmap, levels=10, fill=True, alpha=.6)
    if title: ax.set_title(title, color="#E8EEF6", fontsize=9)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()

def pitch_kde_img(points, xcol, ycol, cmap, title=""):
    """Drop-in for st.pyplot(pitch_kde(...)) -> st.image(pitch_kde_img(...))."""
    if len(points):
        xs = np.ascontiguousarray(points[xcol].to_numpy(dtype="float64"))
        ys = np.ascontiguousarray(points[ycol].to_numpy(dtype="float64"))
    else:
        xs = ys = np.zeros(0, dtype="float64")
    return _pitch_kde_png(xs.tobytes(), ys.tobytes(), cmap, title)


# Cached dual-panel KDE (Tactical Degradation: first 75 vs last 15), same idea.
@st.cache_data(show_spinner=False, max_entries=64)
def _degradation_png(x1b, y1b, x2b, y2b):
    cols = [(np.frombuffer(x1b, dtype="float64"), np.frombuffer(y1b, dtype="float64"), "First 75", "Blues"),
            (np.frombuffer(x2b, dtype="float64"), np.frombuffer(y2b, dtype="float64"), "Last 15", "Reds")]
    fig, axes = plt.subplots(1,2, figsize=(13,5)); fig.patch.set_facecolor("#0E1B33")
    for ax, (xs, ys, ph, cmap) in zip(axes, cols):
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
        if len(xs): pitch.kdeplot(xs, ys, ax=ax, cmap=cmap, levels=8, fill=True, alpha=.55)
        ax.set_title(ph, color="#E8EEF6", fontsize=11)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()

def degradation_img(raw):
    def xy(ph):
        s = raw[raw["phase"] == ph]
        return (np.ascontiguousarray(s["loc_x"].to_numpy(dtype="float64")),
                np.ascontiguousarray(s["loc_y"].to_numpy(dtype="float64")))
    (x1, y1), (x2, y2) = xy("First 75"), xy("Last 15")
    return _degradation_png(x1.tobytes(), y1.tobytes(), x2.tobytes(), y2.tobytes())


# Cached half-pitch KDE on a VerticalPitch (Set Pieces: delivery zones).
@st.cache_data(show_spinner=False, max_entries=64)
def _vhalf_kde_png(xb, yb, cmap):
    xs = np.frombuffer(xb, dtype="float64"); ys = np.frombuffer(yb, dtype="float64")
    fig, ax = plt.subplots(figsize=(8,6)); fig.patch.set_facecolor("#0E1B33")
    pitch = VerticalPitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e", half=True); pitch.draw(ax=ax)
    if len(xs): pitch.kdeplot(xs, ys, ax=ax, cmap=cmap, levels=8, fill=True, alpha=.6)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()

def vhalf_kde_img(points, xcol, ycol, cmap):
    xs = np.ascontiguousarray(points[xcol].to_numpy(dtype="float64"))
    ys = np.ascontiguousarray(points[ycol].to_numpy(dtype="float64"))
    return _vhalf_kde_png(xs.tobytes(), ys.tobytes(), cmap)


def xt_zone_heatmap(grid_player, gx, gy, player_name):
    """Annotated per-zone Expected Threat grid for one player: every cell labelled
    with its value, on top of a real pitch outline, City attacking left to right."""
    nx, ny = len(gx) - 1, len(gy) - 1
    fig, ax = plt.subplots(figsize=(10, 6)); fig.patch.set_facecolor("#0E1B33")
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
    vmax = float(grid_player.max()) if grid_player.max() > 0 else 1.0
    im = ax.imshow(grid_player, extent=[0, 120, 0, 80], origin="lower", cmap="magma",
                   alpha=.88, aspect="auto", zorder=1, vmin=0, vmax=vmax)
    for i in range(ny):
        for j in range(nx):
            x = (gx[j] + gx[j+1]) / 2; y = (gy[i] + gy[i+1]) / 2
            val = grid_player[i, j]
            txt_color = "#1a1a1a" if val > vmax * 0.6 else "white"
            ax.text(x, y, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=7.5, zorder=3)
    ax.set_title(f"{player_name}: Expected Threat added by zone", color="#E8EEF6", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="#E8EEF6")
    for t in cbar.ax.get_yticklabels(): t.set_color("white")
    return fig

def radar_chart(vals_list, names):
    axes = F.RADAR_AXES; fig = go.Figure(); cols=[ui.SKY, ui.GOLD]
    for v,n,c in zip(vals_list, names, cols):
        fig.add_trace(go.Scatterpolar(r=v+[v[0]], theta=axes+[axes[0]], fill="toself",
        name=n, line_color=c, opacity=.55))
    fig.update_layout(polar=dict(bgcolor="#0E1B33",
    radialaxis=dict(visible=True, range=[0,100], color="#5E76A6", gridcolor="#1E2A40")),
    paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EEF6", height=370,
    margin=dict(l=40,r=40,t=24,b=24), legend=dict(orientation="h", y=-.08))
    return fig

def line_chart(df, x, ys, names, title="", xaxis_title="Matchweek", yaxis_title=None, rangeslider=True, colours=None):
    fig = go.Figure(); cols = colours if colours else [ui.SKY, ui.GOLD, ui.CYAN]
    for y,n,c in zip(ys,names,cols):
        fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines+markers", name=n,
        line=dict(color=c,width=3), marker=dict(size=7)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", height=380, title=title,
    margin=dict(l=10,r=10,t=44,b=20),
    xaxis=dict(rangeslider=dict(visible=rangeslider), gridcolor="#1E2A40", title=xaxis_title),
    yaxis=dict(gridcolor="#1E2A40", title=yaxis_title if yaxis_title else (names[0] if len(names)==1 else "")),
    legend=dict(orientation="h", y=-.25))
    return fig


@st.cache_data(show_spinner=False)
def load_xt(_ev):
    return F.build_xt_grid(_ev)
xt_grid, xt_gx, xt_gy = load_xt(city_events)


def _tier_color(ovr):
    return ui.GOLD if ovr >= 82 else (ui.SKY if ovr >= 72 else "#9FB2C8")

def plotly_lineup(xi_rows, formation_name):
    """Interactive lineup placed into proper formation slots."""
    info = {r["player"]: r for r in xi_rows}
    by_group = {"GK":[], "DEF":[], "MID":[], "FWD":[]}
    for r in xi_rows:
        by_group[r["pos_group"]].append(r["player"])
    slots = F.fill_slots(formation_name, by_group)

    fig = go.Figure()
    line = "#3E5C8A"
    fig.add_shape(type="rect", x0=0, y0=0, x1=80, y1=120, line=dict(color=line, width=2), fillcolor="rgba(11,40,28,0.45)")
    fig.add_shape(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color=line, width=1.5))
    fig.add_shape(type="circle", x0=31, y0=51, x1=49, y1=69, line=dict(color=line, width=1.5))
    fig.add_shape(type="rect", x0=22, y0=0, x1=58, y1=16, line=dict(color=line, width=1.5))
    fig.add_shape(type="rect", x0=22, y0=104, x1=58, y1=120, line=dict(color=line, width=1.5))
    fig.add_shape(type="rect", x0=32, y0=0, x1=48, y1=6, line=dict(color=line, width=1.2))
    fig.add_shape(type="rect", x0=32, y0=114, x1=48, y1=120, line=dict(color=line, width=1.2))
    for s in slots:
        x, y = s["x"], s["y"]
        if not s["player"]:
            continue
        r = info[s["player"]]
        ovr = r["ovr"]; fam = r.get("fam",""); role = r.get("role","")
        fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text",
        marker=dict(size=42, color=_tier_color(ovr), line=dict(color="#E8EEF6", width=2), symbol="circle"),
        text=f"<b>{ovr}</b>", textposition="middle center", textfont=dict(color="#06101f", size=13),
        hovertext=f"<b>{r['player']}</b><br>{s['label']} · {role} {fam}<br>OVR {ovr}",
        hoverinfo="text", showlegend=False))
        fig.add_annotation(x=x, y=y-7.5, text=f"{short_name(r['player'])} {fam}", showarrow=False,
        font=dict(color="#E8EEF6", size=10.5, family="Rajdhani"))
        fig.add_annotation(x=x, y=y+7, text=s["label"], showarrow=False,
        font=dict(color="#8FB0DE", size=8))
    fig.update_layout(height=580, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0,r=0,t=6,b=0),
    xaxis=dict(visible=False, range=[-2,82]), yaxis=dict(visible=False, range=[-5,125]))
    return fig


def _add_pitch_shapes_h(fig):
    """Horizontal 120x80 StatsBomb pitch outline as background shapes (City attacks → right)."""
    line = "#3E5C8A"
    fig.add_shape(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line, width=2), fillcolor="rgba(11,40,28,0.45)")
    fig.add_shape(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line, width=1.5))
    fig.add_shape(type="circle", x0=51, y0=31, x1=69, y1=49, line=dict(color=line, width=1.5))
    fig.add_shape(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line, width=1.5))
    fig.add_shape(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line, width=1.5))


def replay_figure(steps, freeze_df):
    """Animated ball/event trail for a possession, with Play/Pause + a scrubber slider,
    real per-segment distance/speed labels, and the genuine shot freeze-frame on the last frame."""
    pts = [steps[0]["start"]] + [s["end"] for s in steps]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mid_x = [(s["start"][0] + s["end"][0]) / 2 for s in steps]
    mid_y = [(s["start"][1] + s["end"][1]) / 2 for s in steps]
    labels = [f"{s['distance_m']:.0f}m" + (f" · {s['speed_kmh']:.0f}km/h" if s["speed_kmh"] else "") for s in steps]

    mates_x = mates_y = opps_x = opps_y = []
    if freeze_df is not None and len(freeze_df):
        mates = freeze_df[freeze_df["teammate"]]; opps = freeze_df[~freeze_df["teammate"]]
        mates_x, mates_y = mates["x"].tolist(), mates["y"].tolist()
        opps_x, opps_y = opps["x"].tolist(), opps["y"].tolist()

    def frame_data(k):
        show_ff = (k == len(steps))
        return [
        go.Scatter(x=xs[:k+1], y=ys[:k+1], mode="lines+markers",
        line=dict(color=ui.GOLD, width=3), marker=dict(size=8, color=ui.SKY), showlegend=False),
        go.Scatter(x=[xs[k]], y=[ys[k]], mode="markers",
        marker=dict(size=15, color="#E8EEF6", line=dict(color=ui.GOLD, width=2)), showlegend=False),
        go.Scatter(x=mid_x[:k], y=mid_y[:k], mode="text", text=labels[:k],
        textfont=dict(color="#E8EEF6", size=10), showlegend=False),
        go.Scatter(x=mates_x if show_ff else [], y=mates_y if show_ff else [], mode="markers",
        marker=dict(size=12, color=ui.SKY, line=dict(color="#E8EEF6", width=1)), name="City", showlegend=show_ff),
        go.Scatter(x=opps_x if show_ff else [], y=opps_y if show_ff else [], mode="markers",
        marker=dict(size=12, color=ui.RED, line=dict(color="#E8EEF6", width=1)), name="Opponent", showlegend=show_ff),
        ]

    n = len(steps)
    frames = [go.Frame(data=frame_data(k), name=str(k)) for k in range(n + 1)]
    fig = go.Figure(data=frame_data(0), frames=frames)
    _add_pitch_shapes_h(fig)
    fig.update_layout(
    height=460, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=6, b=40),
    xaxis=dict(visible=False, range=[-3, 123]), yaxis=dict(visible=False, range=[-5, 85]),
    legend=dict(orientation="h", y=-0.02),
    updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=-0.16, xanchor="left",
    buttons=[
    dict(label="▶ Play", method="animate", args=[None, {"frame": {"duration": 650, "redraw": True},
    "fromcurrent": True, "transition": {"duration": 250}}]),
    dict(label="⏸ Pause", method="animate", args=[[None], {"frame": {"duration": 0, "redraw": False},
    "mode": "immediate"}]),
    ])],
    sliders=[dict(active=0, y=-0.16, x=0.18, len=0.8,
    steps=[dict(method="animate", label=str(k),
    args=[[str(k)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
    for k in range(n + 1)])],
    )
    return fig


def position_picker(grp, n_slots, options, recs, key_prefix, cols):
    """Render n_slots selectboxes for one position group with no two slots able to
    hold the same player: picking a player already sitting in another slot swaps
    that slot to whoever this one is giving up, instead of allowing a duplicate."""
    if not options:
        return []
    slot_keys = [f"{key_prefix}_{grp}{j}" for j in range(n_slots)]
    for i, key in enumerate(slot_keys):
        if key not in st.session_state:
            default = recs[i] if i < len(recs) else options[0]
            st.session_state[key] = default if default in options else options[0]
        st.session_state.setdefault(f"_prev_{key}", st.session_state[key])

    def _make_callback(key, prev_key, slot_keys):
        def _cb():
            new_val = st.session_state[key]
            old_val = st.session_state.get(prev_key, new_val)
            if new_val != old_val:
                for k in slot_keys:
                    if k != key and st.session_state.get(k) == new_val:
                        st.session_state[k] = old_val
                        st.session_state[f"_prev_{k}"] = old_val
                        break
            st.session_state[prev_key] = new_val
        return _cb

    chosen = []
    for i, key in enumerate(slot_keys):
        prev_key = f"_prev_{key}"
        val = cols[i].selectbox(f"{grp}{i+1}", options, key=key, label_visibility="collapsed",
                                 on_change=_make_callback(key, prev_key, slot_keys))
        chosen.append(val)
    return chosen


# ── NAV (grouped accordion menu) ─────────────────────────────────────────────
SECTIONS = {
    "🏟️ Match Centre": ["Dashboard","Squad","Player Profile","Compare Players",
                         "Season Trends","Expected Threat"],
    "🎯 Opponent":     ["Scout Report","Matchup","Who To Stop",
                         "Transition","Tactical Degradation"],
    "⚙️ Selection":    ["Build XI","Availability","Pressing Game","Set Pieces"],
    "🔮 Predict":      ["Season Forecast","Live Assistant"],
    "🧠 Intelligence": ["xG Model","Match Predictor","Form Forecast",
                         "Spatial Lab","Vision Lab"],
    "❓ Help":         ["How To Use"],
}

if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

# ── hide the sidebar — navigation lives in a Safari-style top tab bar ──
st.markdown('<style>[data-testid="stSidebar"],[data-testid="collapsedControl"]{display:none!important;}</style>',
            unsafe_allow_html=True)

_ALL_PAGES = [p for ps in SECTIONS.values() for p in ps]

# header: brand + match controls + search
_hb = st.columns([0.6, 2.7, 1.0, 0.95, 0.42])
_hb[0].image(BADGE, width=68)
_hb[1].markdown('<div class="brand" style="margin-top:2px">Pitch<span>Instinct</span></div>'
                '<div style="font-size:.82rem;color:#9FB1C8;letter-spacing:4px;font-weight:600;margin-top:2px">CITY MANAGER SYSTEM</div>',
                unsafe_allow_html=True)
opponent = _hb[2].selectbox("Opponent", opponents,
                            index=opponents.index("Arsenal") if "Arsenal" in opponents else 0,
                            help="Online demo build: three opponents are preloaded. The full version covers all 19.")
formation_name = _hb[3].selectbox("Formation", list(FORMATIONS.keys()))
formation = FORMATIONS[formation_name]
with _hb[4]:
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)  # align icon with the selectbox inputs
    with st.popover(":material/search:", use_container_width=False):
        st.markdown('<div class="search-sugg-label">Search</div>', unsafe_allow_html=True)
        _q = st.text_input("Search", placeholder="Search pages…", key="globalsearch", label_visibility="collapsed")
        if _q and _q.strip():
            _ql = _q.strip().lower()
            _hits = [p for p in _ALL_PAGES if _ql in p.lower()]
            if _hits:
                for _p in _hits[:8]:
                    if st.button(_p, key=f"sugg_{_p}", use_container_width=True):
                        st.session_state.page = _p
                        st.rerun()
            else:
                st.caption("No matching pages.")

# grouped section dropdowns — one popover per SECTIONS group, holding that group's pages
_nc = st.columns(len(SECTIONS))
for _i, (_sec, _pages) in enumerate(SECTIONS.items()):
    with _nc[_i]:
        _label = _sec + (f"  ·  {st.session_state.page}" if st.session_state.page in _pages else "")
        with st.popover(_label, use_container_width=True):
            for _pg in _pages:
                if st.button(_pg, key=f"pg_{_pg}", use_container_width=True,
                             type=("primary" if _pg == st.session_state.page else "secondary")):
                    st.session_state.page = _pg
                    st.rerun()
st.markdown('<div style="border-bottom:1px solid rgba(255,255,255,.09);margin:8px 0 16px"></div>',
            unsafe_allow_html=True)
page = st.session_state.page

OPP_PAGES = SECTIONS["🎯 Opponent"]
need_scout = page in OPP_PAGES or page in ("Build XI","Dashboard","Live Assistant","Spatial Lab","Squad","Compare Players")
if need_scout:
    with st.spinner(f"Scouting {opponent}..."):
        opp_full = load_opp_full(opponent)
        profile, key_player, fatigue = load_scout(opponent)


# ══════════════════════════════════════════════════════════════════════════════
# MATCH CENTRE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    ui.hero("Welcome back, Pep", f"Manchester City · matchday build-up vs {opponent}", live=True, image=PEP_IMG)
    with st.expander("About this page"):
        st.markdown(
            "Your pre-match snapshot: season form, how both sides are trending, leading scorers, "
            "and the headline threat from the next opponent. For in-game decisions during the "
            "fixture itself, use Predict → Live Assistant.")
    ui.match_banner(OUR_TEAM, opponent, "Next Fixture", "Premier League · Etihad Stadium")

    fx_rows = city_matches[city_matches["opponent"] == opponent].sort_values("match_date")
    cutoff_date = fx_rows.iloc[0]["match_date"] if len(fx_rows) else city_matches["match_date"].max()
    ref_week = int(fx_rows.iloc[0]["match_week"]) if len(fx_rows) else int(city_matches["match_week"].max())
    so_far = city_matches[city_matches["match_date"] < cutoff_date].sort_values("match_week")
    res = so_far["result"].tolist()
    w,d,l = res.count("W"), res.count("D"), res.count("L")

    ov = F.team_overview(city_events, OUR_TEAM)
    ui.section(f"Manchester City's Season So Far (heading into Matchweek {ref_week})")
    st.caption(f"Manchester City's own season numbers, not {opponent}'s.")
    c = st.columns(6)
    c[0].markdown(ui.stat_card("Possession", f"{ov['possession']}%"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Goals / game", ov["goals_pg"]), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Conceded / game", round(so_far["opp_score"].mean(), 1) if len(so_far) else 0.0), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Pass %", f"{ov['pass_pct']}%"), unsafe_allow_html=True)
    c[4].markdown(ui.stat_card("Record", f"{w}-{d}-{l}", "W-D-L so far"), unsafe_allow_html=True)
    with c[5]:
        st.markdown('<div class="tile"><div class="l" style="margin-bottom:6px">Last 5</div>'
        + ui.form_pills(res[-5:]) + '</div>', unsafe_allow_html=True)

    # ── EA-style verdict cards: at-a-glance read of City's profile ──
    _gpg = float(ov["goals_pg"]); _cpg = float(so_far["opp_score"].mean()) if len(so_far) else 0.0
    _poss = float(ov["possession"]); _f5 = res[-5:]; _fpts = _f5.count("W") * 3 + _f5.count("D")
    _atk = ("Elite","elite") if _gpg >= 2 else (("Strong","good") if _gpg >= 1.5 else (("Decent","mid") if _gpg >= 1 else ("Blunt","low")))
    _dfn = ("Solid","elite") if _cpg < 0.9 else (("Steady","good") if _cpg < 1.3 else (("Open","mid") if _cpg < 1.8 else ("Leaky","low")))
    _ctl = ("Dominant","elite") if _poss >= 58 else (("On Top","good") if _poss >= 52 else (("Even","mid") if _poss >= 46 else ("Ceded","low")))
    _frm = ("Hot","elite") if _fpts >= 11 else (("Good","good") if _fpts >= 7 else (("Mixed","mid") if _fpts >= 4 else ("Cold","low")))
    st.markdown("")
    st.caption("At a glance: **Manchester City's** profile so far.")
    vc = st.columns(4)
    for _col, _lbl, (_val, _tone) in zip(vc, ["Attack","Defence","Control","Form"], [_atk,_dfn,_ctl,_frm]):
        _col.markdown(ui.verdict_card(_lbl, _val, _tone), unsafe_allow_html=True)

    st.markdown(""); ui.section("📈 Form Watch")
    table, form_pts = N.season_snapshot(all_pl_matches, cutoff_date)
    city_last5 = res[-5:]
    city_pts5 = city_last5.count("W")*3 + city_last5.count("D")
    opp_pts5 = form_pts.get(opponent)
    fc1, fc2 = st.columns(2)
    with fc1:
        st.markdown(ui.stat_card("Manchester City, last 5", f"{city_pts5} pts", ui.form_pills(city_last5)), unsafe_allow_html=True)
    with fc2:
        if opp_pts5 is not None:
            arrow = "▲" if city_pts5 > opp_pts5 else ("▼" if city_pts5 < opp_pts5 else "＝")
            arrow_color = ui.GREEN if city_pts5 > opp_pts5 else (ui.RED if city_pts5 < opp_pts5 else ui.MUT)
            sub = f'<span style="color:{arrow_color};font-weight:700">{arrow}</span> vs City'
        else:
            sub = ""
        st.markdown(ui.stat_card(f"{opponent}, last 5", f"{opp_pts5} pts" if opp_pts5 is not None else "n/a", sub), unsafe_allow_html=True)
    if opp_pts5 is not None:
        verdict = ("City go in with the better recent form" if city_pts5 > opp_pts5
        else ("It's even on recent form, this one's there to be won" if city_pts5 == opp_pts5
        else f"{opponent} arrive in better form, worth being cautious early on"))
        st.markdown(ui.rec_box("good" if city_pts5 >= (opp_pts5 or 0) else "warn", verdict), unsafe_allow_html=True)

    st.markdown(""); ui.section("⭐ Top Performers")
    top = kpis.copy(); top["ovr"] = top["player"].apply(player_overall)
    top = top.sort_values("ovr", ascending=False).head(8)
    top["Player"] = top["player"].apply(short_name)
    _tpc, _tpd = st.columns([2.4, 1])
    with _tpc:
        _ftp = px.bar(top, x="ovr", y="Player", orientation="h", template="plotly_dark",
                      color="ovr", color_continuous_scale=["#CDEEE8", ui.SKY], text="ovr")
        _ftp.update_traces(textposition="outside", cliponaxis=False)
        _ftp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                           coloraxis_showscale=False, height=360, yaxis=dict(autorange="reversed"),
                           margin=dict(t=10, l=0, r=24, b=10), xaxis_title="Overall rating",
                           yaxis_title="", xaxis=dict(range=[60, 95], gridcolor="#1E2A40"))
        st.plotly_chart(_ftp, use_container_width=True, config={"displayModeBar": False})
    with _tpd:
        _best = top.iloc[0]
        st.markdown(ui.rec_box("info",
            f"<b>Top Performers</b> ranks your squad by <b>overall rating</b> — a FIFA-16 baseline "
            f"adjusted for this season's data. Longer, darker bars are your strongest players. "
            f"<b>{_best['Player']}</b> leads on {int(_best['ovr'])}.", kicker="What this is"),
            unsafe_allow_html=True)

    # ── visual analytics (Season Form · Goal Contributions · Goals For/Against · Results) ──
    _DK = {"displayModeBar": False}
    def _dark(fig, h=300):
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          height=h, margin=dict(t=10, l=0, r=0, b=10),
                          xaxis=dict(gridcolor="#1E2A40"), yaxis=dict(gridcolor="#1E2A40"))
        return fig

    st.markdown("")
    _g1, _g2 = st.columns(2)
    with _g1:
        ui.section("Season Form")
        _cm = city_matches.sort_values("match_week").copy()
        _cm["Points"] = _cm["result"].map({"W":3,"D":1,"L":0}).fillna(0).cumsum()
        _ff = go.Figure(go.Scatter(x=_cm["match_week"], y=_cm["Points"], mode="lines",
              line=dict(color=ui.SKY, width=3), fill="tozeroy", fillcolor="rgba(21,179,160,0.14)"))
        st.plotly_chart(_dark(_ff), use_container_width=True, config=_DK)
    with _g2:
        ui.section("Goal Contributions")
        _gc = kpis.copy(); _gc["GA"] = _gc["goals"] + _gc["assists"]
        _gc = _gc.sort_values("GA", ascending=False).head(8)
        _gc["Player"] = _gc["player"].apply(short_name)
        _fb = px.bar(_gc, x="GA", y="Player", orientation="h", template="plotly_dark",
              color="GA", color_continuous_scale=["#0E1B33", ui.SKY], text="GA")
        _fb.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
              coloraxis_showscale=False, height=300, yaxis=dict(autorange="reversed"),
              margin=dict(t=10, l=0, r=0, b=10), xaxis_title="", yaxis_title="")
        st.plotly_chart(_fb, use_container_width=True, config=_DK)

    _h1, _h2 = st.columns([1.7, 1])
    with _h1:
        ui.section("Goals — For vs Against")
        _cm2 = city_matches.sort_values("match_week").copy()
        _cm2["For"] = _cm2["city_score"].rolling(5, min_periods=1).mean().round(2)
        _cm2["Against"] = _cm2["opp_score"].rolling(5, min_periods=1).mean().round(2)
        _fg = go.Figure()
        _fg.add_trace(go.Scatter(x=_cm2["match_week"], y=_cm2["For"], name="Goals For", mode="lines",
              line=dict(color=ui.SKY, width=3)))
        _fg.add_trace(go.Scatter(x=_cm2["match_week"], y=_cm2["Against"], name="Goals Against", mode="lines",
              line=dict(color=ui.RED, width=3)))
        _fg.update_layout(legend=dict(orientation="h", y=-.2))
        st.plotly_chart(_dark(_fg), use_container_width=True, config=_DK)
    with _h2:
        ui.section("Results")
        _fw = go.Figure(go.Pie(labels=["Win","Draw","Loss"], values=[w,d,l], hole=.62, sort=False,
              marker=dict(colors=[ui.GREEN, ui.AMBER, ui.RED], line=dict(color="#0E1626", width=2)),
              texttemplate="%{label}<br><b>%{value}</b>", textfont=dict(size=13, color="#FFFFFF"),
              hovertemplate="%{label}: %{value}<extra></extra>"))
        _fw.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
              height=300, showlegend=False, margin=dict(t=6, l=0, r=0, b=6))
        st.plotly_chart(_fw, use_container_width=True, config=_DK)

    if need_scout:
        st.markdown(""); ui.section(f"🎯 {opponent}: Headline Threat")
        st.markdown(ui.rec_box("warn",
        f"They play {profile['formation']}, {profile['press_word']}, attack {profile['flank_word']}. "
        f"Key man to stop: <b>{key_player['player'] if key_player else 'n/a'}</b>. "
        f"Full breakdown in the Opponent → Scout Report tab."), unsafe_allow_html=True)

    st.markdown(""); ui.section(f"📰 Season Headlines (heading into matchweek {ref_week})")
    headlines = N.curated_news(cutoff_date, max_items=2) + N.data_storylines(table, form_pts, OUR_TEAM, max_items=2)
    if headlines:
        st.markdown(ui.news_ticker(headlines, tag="Season Watch", as_of=cutoff_date), unsafe_allow_html=True)
        st.caption("Hover to pause. A recap of real, dated storylines and the current table, not a live wire.")
        st.markdown("")
        for kind, h in headlines:
            st.markdown(ui.rec_box("info", h, kicker="History" if kind == "history" else "Live"), unsafe_allow_html=True)
    else:
        st.caption("Nothing notable to report yet this early in the season.")


elif page == "Squad":
    ui.hero("Squad", "Every player at a glance, plus your best XI right now")
    ui.legend("What this shows", [
    "Every Man City player as a card, sorted by overall rating, with a position filter so you can "
    "compare like for like. Below that, a quick best-XI view that reacts to the opponent you've "
    "picked in the sidebar and lets you swap players in and out with the rating updating live.",
    "For a deeper team-selection workspace, with tactical presets, match importance and a "
    "scenario comparer, head to Selection, Build XI.",
    ])
    with st.expander("How these overall ratings are calculated"):
        st.markdown(
        "Ratings are derived, not assumed. Each player starts from a preseason scouting grade, "
        "then that grade is adjusted using their actual 2015/16 output:\n\n"
        "1. A performance score blends six data attributes (shooting, passing, creativity, "
        "pressing, press resistance, defending), weighted by position. A striker leans on "
        "shooting and creativity, a defender on defending, press resistance and passing.\n"
        "2. We measure how far that performance sits above or below the average for the position.\n"
        "3. The adjustment is 0.32 times that gap times a confidence factor, capped at plus or minus 5, "
        "where confidence rises with appearances (so a player with only a handful of games stays close "
        "to the baseline).\n"
        "4. Overall equals the baseline grade plus that adjustment.\n\n"
        "Goalkeepers stay on the baseline since outfield event data cannot fairly rate them. "
        "The result is that over performers rise a couple of points and a player having an "
        "injury hit season dips, all earned from the data rather than fixed.")
    flt = st.radio("Filter", ["All","GK","DEF","MID","FWD"], horizontal=True)
    roster = kpis.copy(); roster["ovr"] = roster["player"].apply(player_overall)
    roster = roster.merge(positions[["player","pos_group"]], on="player", how="left")
    roster["pos_group"] = roster["pos_group"].fillna("MID")
    if flt != "All": roster = roster[roster["pos_group"] == flt]
    roster = roster.sort_values("ovr", ascending=False)
    cols = st.columns(6)
    for i,(_,r) in enumerate(roster.iterrows()):
        cols[i%6].markdown(player_card_html(r["player"]), unsafe_allow_html=True)

    st.markdown(""); ui.section(f"Your Best XI vs {opponent}")
    ov = F.team_overview(city_events, OUR_TEAM)
    c = st.columns(4)
    c[0].markdown(ui.stat_card("Possession", f"{ov['possession']}%", "share of all passes"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Passes / game", int(ov["passes_pg"])), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Shots / game", ov["shots_pg"]), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Pressures / game", int(ov["pressures_pg"]), "work rate"), unsafe_allow_html=True)

    def _suggest_formation(profile):
        if profile["press_level"] == "High":
            return "4-3-3", "an extra body in central midfield helps you play through their high press"
        if profile["leaky"] == "leaky":
            return "4-4-2", "two out and out strikers should punish a defence this open"
        if profile["press_level"] == "Low":
            return "3-5-2", "extra control in midfield to break down a side that sits back"
        return "4-3-3", "a balanced shape that doesn't overcommit either way"
    sugg_form, sugg_why = _suggest_formation(profile)
    st.markdown(ui.rec_box("info", f"Against {opponent}, <b>{sugg_form}</b> looks the better shape: {sugg_why}."), unsafe_allow_html=True)

    fsel = st.selectbox("Formation", list(FORMATIONS.keys()), index=list(FORMATIONS.keys()).index(sugg_form) if sugg_form in FORMATIONS else 0, key="sq_form")
    fdict = FORMATIONS[fsel]
    auto_weights, _ = opponent_priorities(profile)
    scored_q = compute_best_xi_score(squad, pr_ratings, {"press_resistance":auto_weights["press_resistance"],
               "goal_contribution":auto_weights["goal_threat"],"pass_quality":auto_weights["passing"],
               "pressing_output":auto_weights["pressing"]})
    scored_q = scored_q.merge(positions[["player","pos_group","position"]], on="player", how="left")
    scored_q["pos_group"] = scored_q["pos_group"].fillna("MID")
    rec_q = recommend_xi(scored_q, positions, profile, key_player, formation=fdict)

    chosen_q = []
    for grp in ["GK","DEF","MID","FWD"]:
        options = scored_q[scored_q["pos_group"]==grp]["player"].tolist()
        recs = rec_q[rec_q["pos_group"]==grp]["player"].tolist()
        cols_q = st.columns(fdict[grp])
        chosen_q += position_picker(grp, fdict[grp], options, recs,
                                     key_prefix=f"sq_{opponent}_{fsel}", cols=cols_q)

    cur_q = scored_q[scored_q["player"].isin(chosen_q)].merge(
        squad_norm[["player","n_press_res","n_goal","n_creativity","n_passing","n_workrate"]], on="player", how="left")
    team_rating = round(sum(player_overall(pl) for pl in chosen_q) / len(chosen_q)) if chosen_q else 0
    xi_rows = []
    for pl in chosen_q:
        r = cur_q[cur_q["player"]==pl]
        grp = r.iloc[0]["pos_group"] if len(r) else "MID"
        role = F.assign_role(r.iloc[0]) if len(r) else ""
        fam = F.role_familiarity(r.iloc[0]) if len(r) else ""
        xi_rows.append({"player": pl, "pos_group": grp, "ovr": player_overall(pl), "role": role, "fam": fam})
    cc1, cc2 = st.columns([1,3])
    cc1.markdown(ui.stat_card("Team rating", team_rating, "average overall, live"), unsafe_allow_html=True)
    with cc2:
        st.caption("Swap any player above. The pitch and the team rating update with your picks.")
    st.plotly_chart(plotly_lineup(xi_rows, fsel), use_container_width=True)


elif page == "Player Profile":
    ui.hero("Player Profile", "Card, attributes, season KPIs and form trend")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Player Profile"])
    ui.legend("What this shows", [
    "Pick any City player for the full picture: their card, six data-derived attribute "
    "bars (0-100), this season's headline KPIs, and how the chosen metric has trended match by match.",
    "The \"how this overall is derived\" panel shows the exact maths behind that player's specific rating.",
    ])
    pl = st.selectbox("Select player", sorted(kpis["player"].tolist()),
    index=sorted(kpis["player"].tolist()).index("Kevin De Bruyne")
    if "Kevin De Bruyne" in kpis["player"].values else 0)
    krow = kpis[kpis["player"] == pl].iloc[0]
    attrs, ovr = F.player_attributes(krow, attr_ranges)

    col1, col2 = st.columns([1,1.6])
    with col1:
        st.markdown(player_card_html(pl), unsafe_allow_html=True)
    with col2:
        ui.section("Attributes")
        for k,v in attrs.items():
            st.markdown(ui.attr_bar(k, v), unsafe_allow_html=True)

            # transparent rating derivation for THIS player
    od = OVERALLS.get(pl)
    if od:
        ui.section("How this overall is derived")
        base_txt = f"{od['base']}" if od["base"] is not None else "positional baseline"
        if od["pos"] == "GK":
            calc = f"Goalkeeper, anchored to the preseason baseline ({base_txt})."
        else:
            sign = "+" if od["adj"] >= 0 else ""
            calc = (f"Preseason baseline <b>{base_txt}</b> &nbsp;+&nbsp; adjustment "
            f"<b>{sign}{od['adj']}</b> &nbsp;=&nbsp; <b>{od['overall']}</b><br>"
            f"<span style='color:#9FB6DA;font-size:.85rem'>"
            f"performance {od['perf']} · vs-peers delta {od['delta']:+} · "
            f"confidence {od['conf']} (from {int(krow['apps'])} apps) · "
            f"adj = 0.32 × delta × confidence, capped ±5</span>")
        st.markdown(f'<div class="card">{calc}</div>', unsafe_allow_html=True)

    st.markdown(""); ui.section("Season KPIs")
    c = st.columns(5)
    c[0].markdown(ui.stat_card("Apps", int(krow["apps"])), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Goals", int(krow["goals"])), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Assists", int(krow["assists"])), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Pass %", f"{krow['pass_pct']}%"), unsafe_allow_html=True)
    c[4].markdown(ui.stat_card("Pressures", int(krow["pressures"])), unsafe_allow_html=True)

    st.markdown(""); ui.section("Form Trend & Where It Happens")
    metric = st.selectbox("Metric", ["pass_pct","goals","shots","pressures","passes"])
    metric_label = metric.replace("_"," ").title()
    fcol1, fcol2 = st.columns([1.3, 1])
    with fcol1:
        tr = F.player_trend(city_events, city_matches, pl, OUR_TEAM)
        if len(tr):
            st.plotly_chart(line_chart(tr, "match_week", [metric], [metric_label],
            yaxis_title=metric_label), use_container_width=True)
            st.caption(f"{short_name(pl)}'s {metric_label.lower()} by matchweek this season. "
            "Drag the range slider underneath to zoom into a stretch of games.")
        else:
            st.caption("Not enough matches yet to plot a trend.")
    with fcol2:
        pts, mode = player_metric_points(city_events, pl, metric)
        if mode == "success" and len(pts):
            fig, ax = plt.subplots(figsize=(6.5,5)); fig.patch.set_facecolor("#0E1B33")
            pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
            ok, no = pts[pts["success"]], pts[~pts["success"]]
            pitch.scatter(ok["loc_x"], ok["loc_y"], ax=ax, s=55, color=ui.GREEN, alpha=.8, edgecolors="white", label="Completed", zorder=4)
            pitch.scatter(no["loc_x"], no["loc_y"], ax=ax, s=55, color=ui.RED, alpha=.8, edgecolors="white", label="Lost", zorder=4)
            ax.legend(facecolor="#101D3D", labelcolor="#E8EEF6", loc="upper left", fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()
            st.caption(f"Every pass {short_name(pl)} has attempted this season, green where it found a teammate.")
        elif len(pts):
            st.image(pitch_kde_img(pts, "loc_x", "loc_y", "Blues" if metric != "goals" else "Oranges"), use_container_width=True)
            st.caption(f"Where {short_name(pl)}'s {metric_label.lower()} actually happen on the pitch this season.")
        else:
            st.info(f"No located {metric_label.lower()} events for {short_name(pl)} yet.")


elif page == "Compare Players":
    ui.hero("Compare Players", "Two City players head-to-head")
    ui.legend("What this shows", [
    "Two players side by side as cards, a radar overlaying both on the same five normalised axes "
    "(press resistance, goal threat, creativity, passing, work rate), and a read on how each of "
    f"them is best used specifically against {opponent}.",
    ])
    players = sorted(kpis["player"].tolist())
    a,b = st.columns(2)
    p1 = a.selectbox("Player A", players, index=players.index("Kevin De Bruyne") if "Kevin De Bruyne" in players else 0)
    p2 = b.selectbox("Player B", players, index=players.index("David Josué Jiménez Silva") if "David Josué Jiménez Silva" in players else 1)
    ca, cb = st.columns(2)
    ca.markdown(player_card_html(p1), unsafe_allow_html=True)
    cb.markdown(player_card_html(p2), unsafe_allow_html=True)

    r1 = squad_norm[squad_norm["player"]==p1].iloc[0]; r2 = squad_norm[squad_norm["player"]==p2].iloc[0]
    def vec(r): return [r["n_press_res"],r["n_goal"],r["n_creativity"],r["n_passing"],r["n_workrate"]]
    st.markdown(""); ui.section("Profile Overlay")
    st.plotly_chart(radar_chart([vec(r1),vec(r2)],[p1.split()[-1],p2.split()[-1]]), use_container_width=True)
    st.caption("Each axis is normalised 0-100 against the rest of the squad, so this shows relative strengths, not raw totals.")

    def _usage_tip(name, r):
        tips = []
        if profile["press_level"] == "High" and r["n_press_res"] >= 65:
            tips.append("composed enough on the ball to play through their high press")
        if profile["leaky"] == "leaky" and r["n_goal"] >= 65:
            tips.append("enough of a goal threat to punish a defence this open")
        if profile["press_level"] == "Low" and r["n_passing"] >= 65:
            tips.append("the range of passing to help break down a side that sits back")
        if r["n_workrate"] >= 70:
            tips.append("the work rate to press them high if needed")
        if r["n_creativity"] >= 70:
            tips.append("enough creativity to unlock a settled defence")
        if not tips:
            return f"{short_name(name)} is a balanced, low-risk option here without a standout edge for this specific matchup."
        return f"Against {opponent}, {short_name(name)} brings " + ", and ".join(tips) + "."

    st.markdown(""); ui.section(f"How to use them vs {opponent}")
    uc1, uc2 = st.columns(2)
    uc1.markdown(ui.rec_box("good", _usage_tip(p1, r1)), unsafe_allow_html=True)
    uc2.markdown(ui.rec_box("good", _usage_tip(p2, r2)), unsafe_allow_html=True)

    st.markdown(""); ui.section("Where Their Goal Threat Comes From")
    mc1, mc2 = st.columns(2)
    for col, pl_, nm in [(mc1, p1, p1), (mc2, p2, p2)]:
        pts, _ = player_metric_points(city_events, pl_, "shots")
        with col:
            if len(pts):
                st.image(pitch_kde_img(pts, "loc_x", "loc_y", "Oranges", short_name(nm)), use_container_width=True)
            else:
                st.info(f"No located shots for {short_name(nm)} this season.")


elif page == "Season Trends":
    ui.hero("Performance Trends", "Track team form across the season, and catch dips before they cost points")
    ui.legend("What this shows", [
    "Points accumulated matchweek by matchweek, and a rolling 5-game average of goals for versus "
    "against. The rolling view smooths out single-game noise so a genuine hot streak or a "
    "defensive dip stands out from normal week-to-week variation.",
    "Below the charts, the worst patch of form this season is picked out automatically, with what "
    "was actually happening in it and a suggestion for guarding against the same thing again.",
    ])
    cm = city_matches.sort_values("match_week").copy()
    cm["goals_for"] = cm["city_score"]; cm["goals_against"] = cm["opp_score"]
    cm["pts"] = cm["result"].map({"W": 3, "D": 1, "L": 0})
    if "points_running" in cm.columns:
        ui.section("Points Accumulation")
        st.plotly_chart(line_chart(cm, "match_week", ["points_running"], ["Points"],
                        yaxis_title="Points", rangeslider=False), use_container_width=True)
        ppg = round(cm["pts"].mean(), 2)
        projected = round(ppg * 38)
        st.caption(f"Cumulative points across the season. A flattening line is a stalled run, a steepening one is "
                   f"momentum building. Current pace is {ppg} points per game, which projects to roughly "
                   f"{projected} points over a full 38-game season.")

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    ui.section("Goals For vs Against (rolling)")
    cm["gf_roll"] = cm["goals_for"].rolling(5, min_periods=1).mean().round(2)
    cm["ga_roll"] = cm["goals_against"].rolling(5, min_periods=1).mean().round(2)
    st.plotly_chart(line_chart(cm, "match_week", ["gf_roll","ga_roll"], ["Goals For (5-game avg)","Goals Against (5-game avg)"],
                    yaxis_title="Goals per game", rangeslider=False, colours=[ui.SKY, ui.RED]), use_container_width=True)
    latest_gf, latest_ga = cm["gf_roll"].iloc[-1], cm["ga_roll"].iloc[-1]
    if latest_gf > latest_ga:
        trend_txt = f"Right now the blue line sits above the red ({latest_gf} vs {latest_ga}), recent form is net positive in front of goal."
    elif latest_ga > latest_gf:
        trend_txt = f"Right now the red line sits above the blue ({latest_ga} vs {latest_gf}), the team is currently conceding more than it scores over its last 5."
    else:
        trend_txt = f"The two lines are level at {latest_gf}, an evenly balanced recent run either way."
    st.caption(f"Five-game rolling average. {trend_txt}")

    st.markdown("<div style='height:36px'></div>", unsafe_allow_html=True)
    ui.section("📉 Worst Patch Of Form")
    dip = F.detect_dip(city_matches)
    if dip:
        c = st.columns(3)
        c[0].markdown(ui.stat_card("Window", f"MW{dip['start_week']}-{dip['end_week']}"), unsafe_allow_html=True)
        c[1].markdown(ui.stat_card("Form in it", f"{dip['dip_avg_pts']} pts/g", f"season avg {dip['season_avg_pts']}"), unsafe_allow_html=True)
        c[2].markdown(ui.stat_card("Scored vs Conceded", f"{dip['gf']}-{dip['ga']}"), unsafe_allow_html=True)
        st.markdown(f"<div class='card'>Results in that window: {ui.form_pills(dip['results'])} against "
        f"{', '.join(dip['opponents'])} ({dip['home_n']} at home, {dip['away_n']} away).</div>", unsafe_allow_html=True)

        if dip["ga"] > dip["gf"] * 1.5:
            cause = "the defence leaked goals faster than the attack could keep pace, the clearest sign of a structural defensive issue rather than bad luck."
            fix = "Check Availability for fatigue among your back line over that period, and lean on Pressing Game to see if the press was getting bypassed."
        elif dip["gf"] <= 2:
            cause = "the goals dried up, the team kept results competitive defensively but couldn't find the net."
            fix = "Lean the Build XI weighting toward goal threat for the next few fixtures, or look at Set Pieces as an extra route to goal."
        elif dip["away_n"] >= 3:
            cause = "it came against a tough run of mostly away fixtures."
            fix = "Expect form to recover at home; this looks more like fixture difficulty than a real drop in quality."
        else:
            cause = "a mix of stretched form across a packed run of fixtures."
            fix = "Worth checking Availability for rotation candidates before a run like this comes round again."
        st.markdown(ui.rec_box("warn", f"What happened: {cause}"), unsafe_allow_html=True)
        st.markdown(ui.rec_box("good", f"Guarding against it: {fix}"), unsafe_allow_html=True)
    else:
        st.caption("Not enough matches played yet to identify a dip.")


        # ══════════════════════════════════════════════════════════════════════════════
        # OPPONENT
        # ══════════════════════════════════════════════════════════════════════════════
elif page == "Scout Report":
    ui.hero(f"Scouting Report: {opponent}", f"They line up {profile['formation']} · full-season profile")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Scout Report"])
    ui.legend("What this shows", [
    f"{opponent}'s full-season profile built from their real match data: how hard/high they press, "
    "which side they attack down, how potent and how leaky they are, and their usual formation.",
    "Tabs below break it down further, their danger men, likely starting XI, where they win/lose "
    "the ball, set-piece threat, and how their pressing changes by game state.",
    ])
    S = load_scout_report(opponent)
    brief = (f"{opponent} play {profile['press_word']} ({profile['press_pm']} pressures/game), attack mostly "
    f"{profile['flank_word']}, and are a {profile['potent']} attack ({profile['xg_pm']} xG/game). "
    f"Defensively they are {profile['leaky']} ({profile['goals_conc_pm']} conceded/game).")
    st.markdown(f'<div class="card"><b style="color:{ui.SKY}">Brief:</b> {brief}</div>', unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(ui.stat_card("Press", profile["press_level"], f"{profile['press_pm']}/g"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Attack", profile["flank"], f"{profile['xg_pm']} xG/g"), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Defence", profile["leaky"].title(), f"{profile['goals_conc_pm']}/g"), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Shape", profile["formation"]), unsafe_allow_html=True)

    tabs = st.tabs(["⚠️ Danger Men","📋 Likely XI","🎯 Pressing","🔵 Set Pieces","⏱️ Game-State","🗺️ Maps"])
    with tabs[0]:
        dm = S["danger"]
        if len(dm):
            st.caption("G = goals, A = assists, xG = the quality of chances they've had, KP = key passes (the pass before a shot).")
            cols = st.columns(min(len(dm),5))
            for i,(_,r) in enumerate(dm.iterrows()):
                cols[i].markdown(f'<div class="tile"><div style="font-family:Rajdhani;font-weight:700;color:{ui.SKY}">{r["player"].split()[-1]}</div>'
                f'<div class="v" style="font-size:1.5rem">{int(r["goals"])}<span style="font-size:.6rem">G</span> {int(r["assists"])}<span style="font-size:.6rem">A</span></div>'
                f'<div class="s">{r["xg"]} xG · {int(r["key_passes"])} KP</div></div>', unsafe_allow_html=True)
            st.markdown("")
            for _, r in dm.head(3).iterrows():
                nm = r["player"].split()[-1]
                if r["goals"] >= r["assists"] and r["goals"] > 0:
                    threat_txt = f"finishes his own chances, {int(r['goals'])} goals from {r['xg']} xG"
                    fix_txt = "deny him space in the box rather than the ball at his feet"
                elif r["assists"] > 0 or r["key_passes"] >= 5:
                    threat_txt = f"creates more than he scores, {int(r['key_passes'])} key passes feeding others"
                    fix_txt = "press him early so the supply line never gets going"
                else:
                    threat_txt = f"a {r['xg']} xG threat without the numbers on the scoresheet yet"
                    fix_txt = "don't switch off just because the goal tally is low, the chances are there"
                st.markdown(ui.rec_box("warn", f"<b>{nm}</b> {threat_txt}. Contain by: {fix_txt}."), unsafe_allow_html=True)
    with tabs[1]:
        lx = S["likely_xi"]
        if len(lx):
            lx2 = lx.copy(); lx2.index = range(1,len(lx2)+1)
            st.dataframe(lx2[["player","position","starts"]].rename(columns={"player":"Player","position":"Position","starts":"Starts"}), use_container_width=True, height=420)
            attackers = lx[lx["position"].str.contains("Forward|Wing|Striker", case=False, na=False)]["player"].tolist()
            if attackers:
                st.markdown(ui.rec_box("info",
                "Their first-choice attack looks like <b>" + ", ".join(n.split()[-1] for n in attackers) +
                "</b>. Brief your back line on these names specifically and make sure Build XI accounts for "
                f"the same number of attacking outlets when you settle on a formation against {opponent}."), unsafe_allow_html=True)
    with tabs[2]:
        pt = S["pressing"]; col1,col2 = st.columns(2)
        with col1: st.image(pitch_kde_img(pt["win_points"],"loc_x","loc_y","Greens","Where THEY win it"), use_container_width=True)
        with col2: st.image(pitch_kde_img(pt["lose_points"],"loc_x","loc_y","Reds","Where THEY lose it"), use_container_width=True)
        lose_zone = pt["lose_third"].idxmax() if len(pt["lose_third"]) else "Middle"
        st.markdown(ui.rec_box("info", f"They win the ball mainly in the <b>{pt['win_zone']} third</b>, and give it away "
        f"most often in their <b>{lose_zone.lower()} third</b>."), unsafe_allow_html=True)
        st.markdown(ui.rec_box("good", f"That losing zone is exactly where to press them, win it in their "
        f"{lose_zone.lower()} third and you're already in a dangerous area."), unsafe_allow_html=True)
    with tabs[3]:
        sp = S["set_piece"]; c = st.columns(2)
        c[0].markdown(ui.stat_card("Their SP threat", f"{sp['threat_pg']}/g", f"{sp['threat_goals']} goals"), unsafe_allow_html=True)
        c[1].markdown(ui.stat_card("Their SP weakness", f"{sp['vuln_pg']}/g", f"{sp['vuln_shots']} allowed"), unsafe_allow_html=True)
        st.caption("Threat = shots they generate per game from corners, free kicks and throw-ins. Weakness = shots they concede per game from the same situations.")
        if sp["vuln_pg"] > sp["threat_pg"] * 1.2:
            st.markdown(ui.rec_box("good", "They give up more from dead balls than they create, push extra bodies forward on your own corners and free kicks."), unsafe_allow_html=True)
        elif sp["threat_pg"] > sp["vuln_pg"] * 1.2:
            st.markdown(ui.rec_box("warn", "Dead balls are a real weapon for them, assign clear marking on your near and far post for every corner they win."), unsafe_allow_html=True)
        else:
            st.markdown(ui.rec_box("info", "Set pieces are roughly a wash either way for them, not the deciding factor in this game."), unsafe_allow_html=True)
    with tabs[4]:
        gs = S["game_state"]; c = st.columns(3)
        c[0].markdown(ui.stat_card("Leading", gs.get("Leading",0)), unsafe_allow_html=True)
        c[1].markdown(ui.stat_card("Level", gs.get("Level",0)), unsafe_allow_html=True)
        c[2].markdown(ui.stat_card("Trailing", gs.get("Trailing",0)), unsafe_allow_html=True)
        st.caption("Pressures per minute while leading, level or trailing, how hard they chase the ball changes with the scoreline.")
        if gs:
            hi = max(gs, key=gs.get); lo = min(gs, key=gs.get)
            if gs[hi] > 0:
                st.markdown(ui.rec_box("info", f"They press hardest while <b>{hi.lower()}</b> and ease off the most while "
                f"<b>{lo.lower()}</b>. If you can get the game into a {lo.lower()} state for them, you'll see more of the ball."), unsafe_allow_html=True)
    with tabs[5]:
        col1,col2 = st.columns(2)
        with col1:
            ui.section("How they press")
            st.image(pitch_kde_img(profile["pressure_points"],"loc_x","loc_y","Oranges"), use_container_width=True)
            st.caption("Brighter patches are where they apply pressure most often, build-up through the quiet areas instead.")
        with col2:
            ui.section("Where they concede")
            st.image(pitch_kde_img(profile["conceded_points"],"def_x","def_y","Reds"), use_container_width=True)
            st.caption("Brighter patches are where shots against them cluster, that's where your forwards should look to get on the end of things.")


elif page == "Matchup":
    ui.hero(f"Matchup: City ({formation_name}) vs {opponent} ({profile['formation']})", "What decides this game and where the shape battle is won")
    ui.legend("What this shows", [
    f"Everything you need for the tactical build-up against {opponent} in one place: the handful "
    "of factors most likely to decide the game, your shape against theirs, a projected scoreline you "
    "can tilt with the approach slider, and the individual duels your players will face.",
    ])
    ui.section("What decides this game")
    xf = F.x_factors(profile, load_setpiece(opponent), key_player, load_gamestate(opponent), fatigue)
    for kind, title, desc in xf:
        col = ui.RED if kind=="danger" else ui.GREEN; icon = "🛑" if kind=="danger" else "✅"
        st.markdown(f'<div class="card" style="border-left:5px solid {col}"><div style="font-family:Rajdhani;font-weight:700;color:{col};font-size:1.1rem">{icon} {title}</div><div style="margin-top:4px">{desc}</div></div>', unsafe_allow_html=True)
    st.markdown(""); ui.section("⚔️ Battlegrounds")
    for tag, txt in battlegrounds(profile, key_player, fatigue):
        st.markdown(ui.rec_box("warn" if tag=="threat" else "good", txt.replace("**","")), unsafe_allow_html=True)

    st.markdown(""); ui.section("Approach and projection")
    risk = st.slider("⚖️ Your approach into this game", -1.0, 1.0, 0.0, step=0.5,
    help="Drag toward Attacking to open the game up, or Defensive to shut it down, the projection below reacts live.")
    risk_label = {-1.0: "Defensive", -0.5: "Cautious", 0.0: "Balanced", 0.5: "Attacking", 1.0: "All-out Attack"}[risk]
    st.caption(f"Approach: **{risk_label}**")
    proj = F.projected_score(city_events, city_matches, opp_full, opponent, risk=risk)
    c = st.columns(4)
    c[0].markdown(ui.stat_card("Projected", proj["scoreline"], "expected goals"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Win", f"{proj['p_win']}%"), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Draw", f"{proj['p_draw']}%"), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Lose", f"{proj['p_lose']}%"), unsafe_allow_html=True)
    st.caption("xG blend of each attack vs the other defence, tilted by your chosen approach; W/D/L via Poisson (descriptive, not ML).")
    if risk > 0:
        st.markdown(ui.rec_box("warn" if risk >= 0.5 else "info",
        f"Going {risk_label.lower()} raises both sides' chances, it boosts your win odds here but also "
        f"raises {opponent}'s, especially if their attack ({profile['potent']}) punishes the extra space."), unsafe_allow_html=True)
    elif risk < 0:
        st.markdown(ui.rec_box("info",
        f"Going {risk_label.lower()} suppresses chances both ways, sensible if you rate your defence over "
        f"your ability to break down a {profile['leaky']} side."), unsafe_allow_html=True)
    st.markdown(""); ui.section("Shape vs Shape")
    fig, ax = plt.subplots(figsize=(11,5.5)); fig.patch.set_facecolor("#0E1B33")
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
    for pt in F.formation_xy(formation_name.replace("-",""), "home"):
        ax.scatter(pt[0],pt[1], s=440, color=ui.SKY, edgecolors="white", zorder=4)
    for pt in F.formation_xy(profile["formation"], "away"):
        ax.scatter(pt[0],pt[1], s=440, color=ui.RED, edgecolors="white", zorder=4)
    ax.text(27,76,"Man City", color=ui.SKY, fontsize=11, weight="bold", ha="center")
    ax.text(93,76,opponent, color=ui.RED, fontsize=11, weight="bold", ha="center")
    st.pyplot(fig, use_container_width=True); plt.close()
    for tag, txt in F.formation_mismatches(formation_name.replace("-",""), profile["formation"]):
        st.markdown(ui.rec_box("warn" if tag=="warn" else "good", txt), unsafe_allow_html=True)
    st.markdown(""); ui.section("🥊 Key Duels")
    scored = compute_best_xi_score(squad, pr_ratings, {"press_resistance":.3,"goal_contribution":.3,"pass_quality":.2,"pressing_output":.2})
    xi_now = recommend_xi(scored, positions, profile, key_player, formation=formation)
    for d in F.key_duels(load_danger(opponent), opp_full, opponent, xi_now, positions):
        marker = d["marker"] or "your defender"
        st.markdown(f'<div class="card"><b style="color:{ui.GOLD}">{d["threat"]}</b> ({int(d["goals"])}G {int(d["assists"])}A, attacks {d["channel"]}) <span style="color:#7E96C4">vs</span> <b style="color:{ui.SKY}">{marker}</b></div>', unsafe_allow_html=True)


elif page == "Who To Stop":
    ui.hero(f"Who To Stop: {opponent}", "Their creative hub. Cut his supply and the build-up breaks down.")
    ui.legend("What this shows", [
    "We rebuild the opponent's <b>passing network</b> (who passes to whom) and rank players by <b>betweenness centrality</b>, how much of their play flows through each man (lecture: ranking / PageRank family).",
    "<b>Goalkeepers are excluded</b> and the headline target prioritises midfielders/playmakers, the men who actually orchestrate attacks.",
    "On the pitch, <b>bigger gold nodes = more central</b>; thicker lines = more frequent pass combinations.",
    ])
    cent, edges, ranked, hub = load_who_to_stop(opponent)
    if hub is not None:
        st.markdown(f"<div class='card'>We rebuild {opponent}'s passing network from every completed pass this season, "
        "who passes to whom, and how often. The node every pass seems to run through, whether or not he "
        "scores or assists himself, is the one worth marking out of the game.</div>", unsafe_allow_html=True)
        c = st.columns(3)
        c[0].markdown(ui.stat_card("Stop This Man", hub["player"].split()[-1], hub["position"]), unsafe_allow_html=True)
        c[1].markdown(ui.stat_card("Centrality", f"{hub['betweenness_centrality']:.3f}"), unsafe_allow_html=True)
        c[2].markdown(ui.stat_card("Involvement", int(hub["total_involvement"]), "pass actions"), unsafe_allow_html=True)
        ui.section("Passing Network (goalkeepers removed)")
        st.caption("Showing the strongest passing combinations only, so the pitch stays readable. Gold node size is centrality, bigger means more of their play runs through that player; thicker lines are more frequent combinations; only the most central players are labelled.")
        fig, ax = plt.subplots(figsize=(9,6)); fig.patch.set_facecolor("#0E1B33")
        pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
        gk_names = set(cent["player"]) - set(ranked["player"])
        if len(edges):
            edges_o = edges[~edges["player"].isin(gk_names) & ~edges["pass_recipient"].isin(gk_names)]
            edges_o = edges_o.nlargest(min(18, len(edges_o)), "count")  # only the strongest combinations, to keep the pitch readable
            mx = edges_o["count"].max() if len(edges_o) else 1
            for _,e in edges_o.iterrows():
                if pd.isna(e["src_x"]): continue
                ax.plot([e["src_x"],e["tgt_x"]],[e["src_y"],e["tgt_y"]], lw=0.6+2*(e["count"]/mx), color=ui.SKY, alpha=.15+.35*(e["count"]/mx), zorder=2)
            nodes = edges_o.groupby("player").agg(x=("src_x","mean"),y=("src_y","mean")).reset_index().merge(ranked[["player","betweenness_centrality"]], on="player", how="left")
            nodes["betweenness_centrality"] = nodes["betweenness_centrality"].fillna(0)
            pitch.scatter(nodes["x"],nodes["y"], s=150+nodes["betweenness_centrality"]*2500, color=ui.GOLD, edgecolors="white", zorder=4, ax=ax)
            labelled = nodes.nlargest(min(8, len(nodes)), "betweenness_centrality")  # only label the most central players
            for _,n in labelled.iterrows(): ax.text(n["x"],n["y"]+2.5, n["player"].split()[-1], color="#E8EEF6", fontsize=7, ha="center", zorder=5)
        st.pyplot(fig, use_container_width=True); plt.close()
        ui.section("Centrality Ranking (outfield)")
        st.caption("Betweenness centrality: how often a player sits on the shortest passing path between two teammates. "
        "A high score means a lot of their play has to go through him to connect defence to attack.")
        rr = ranked.head(8)[["player","position","betweenness_centrality","total_involvement"]].copy()
        rr.columns = ["Player","Position","Centrality","Involvement"]; rr.index = range(1,len(rr)+1)
        st.dataframe(rr.style.background_gradient(subset=["Centrality"], cmap="Oranges"), use_container_width=True)
        st.markdown(ui.rec_box("good", f"<b>Stop {hub['player']}</b> ({hub['position']}), their play runs through him. Assign your nearest midfielder to deny him time and the attack stalls."), unsafe_allow_html=True)


elif page == "Transition":
    ui.hero(f"Transition Vulnerability: {opponent}", "When we win it back, where is their space?")
    ui.legend("What this shows", [
    f"Where {opponent}'s players are positioned in the first few seconds right after THEY lose the "
    "ball, that's the space City can break into immediately on a turnover, before their shape resets.",
    "Slide the window to widen or tighten how many seconds after the turnover count as 'transition'.",
    ])
    win = st.slider("Seconds after winning the ball", 3, 8, 5)
    trans = load_transition(opponent, win)
    if len(trans):
        c = st.columns(2)
        c[0].markdown(ui.stat_card("Events mapped", len(trans)), unsafe_allow_html=True)
        c[1].markdown(ui.stat_card("Turnovers", trans["match_id"].nunique()), unsafe_allow_html=True)
        st.image(pitch_kde_img(trans,"loc_x","loc_y","Reds", f"{opponent} positions in first {win}s after losing the ball"), use_container_width=True)
        st.caption("Dark patches are where their players actually stand in that window, pale patches are space they've left open.")

        loc = trans.dropna(subset=["loc_y"])
        bands = {"left wing": (loc["loc_y"] < 26.7).sum(), "centre": ((loc["loc_y"] >= 26.7) & (loc["loc_y"] < 53.3)).sum(),
        "right wing": (loc["loc_y"] >= 53.3).sum()}
        total = sum(bands.values())
        if total:
            emptiest = min(bands, key=bands.get)
            share = bands[emptiest] / total * 100
            st.markdown(ui.rec_box("good", f"Right now their {emptiest} is the thinnest covered, only "
            f"{share:.0f}% of their recovery shape sits there in the first {win}s. That's the channel to break "
            f"into the instant you win the ball back, before it fills in."), unsafe_allow_html=True)
    else:
        st.warning("Not enough transition data for this opponent.")


elif page == "Tactical Degradation":
    ui.hero(f"Tactical Degradation: {opponent}", "Does their shape collapse late?")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Tactical Degradation"])
    ui.legend("What this shows", [
    f"Compares {opponent}'s defensive shape in the first 75 minutes vs. the last 15, a deeper, "
    "wider block late on is a fatigue/concentration signal for exactly when to push hardest.",
    ])
    td = load_degradation(opponent)
    st.caption("Depth shift compares where their defensive actions (tackles, blocks, clearances, pressures) happen on "
    "average in the first 75 minutes versus the last 15. A positive number means those actions are "
    "happening deeper, closer to their own goal, late on, a fatigue or concentration signal.")
    c = st.columns(2)
    c[0].markdown(ui.stat_card("Depth shift (last 15)", f"{td['x_shift']:+.1f}", "+ = deeper"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Lateral drift (last 15)", f"{td['y_shift']:+.1f}", "sideways shift in average position"), unsafe_allow_html=True)
    raw = td["raw_events"].dropna(subset=["loc_x","loc_y"])
    st.image(degradation_img(raw), use_container_width=True)
    st.caption("Blue is their defensive shape across the first 75 minutes, red is the last 15. A patch that's shifted "
    "noticeably toward their own goal in red is the late fatigue window.")
    if td["x_shift"] > 3:
        st.markdown(ui.rec_box("good", f"They drop {abs(td['x_shift']):.0f} yards deeper late on, the map above backs it up "
        "with red sitting closer to their own goal than blue. Save a fresh runner for after the 75th minute and "
        "go directly at the space they're conceding."), unsafe_allow_html=True)
    else:
        st.markdown(ui.rec_box("info", "Their shape stays stable late, the blue and red patches sit in roughly the same "
        "place, so there's no obvious fatigue window to target here. Decide the game with quality, not a late surge."), unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════════════
        # SELECTION
        # ══════════════════════════════════════════════════════════════════════════════
elif page == "Build XI":
    ui.hero(f"Build Your XI vs {opponent}", f"{formation_name} · auto-picked to beat them, fully editable")
    ui.legend("What this shows", [
    "An auto-recommended XI tuned to this specific opponent, pick a tactical preset and a match "
    "importance, then swap any player; the lineup, balance radar and roles all update live.",
    "The Scenario Comparer at the bottom lets you pressure-test the recommendation against a second style before committing.",
    ])

    # one-click tactical preset, overrides the auto weights if chosen
    pc1, pc2 = st.columns([1.4, 1])
    preset_name = pc1.selectbox("⚡ Tactical Preset", list(F.TACTICAL_PRESETS.keys()),
    help="Auto is tuned to this opponent. The others emulate a manager style.")
    importance = pc2.radio("🏆 Match importance", ["Balanced", "Must-Win", "Rotation"], horizontal=True,
    help="Must-Win leans harder into goal threat & press resistance. Rotation downweights your most heavily-loaded players.")
    auto_weights, reasons = opponent_priorities(profile)
    preset = F.TACTICAL_PRESETS[preset_name]
    if preset is None:
        weights = dict(auto_weights)
        why = "Because " + "; ".join(r.replace("**","") for r in reasons) + "." if reasons else "Balanced selection."
    else:
        weights = dict(preset)
        why = f"<b>{preset_name}</b> preset selected, prioritising " + \
              max(preset, key=preset.get).replace("_"," ") + "."

    if importance == "Must-Win":
        weights["goal_threat"] = weights.get("goal_threat", 0.3) * 1.4
        weights["press_resistance"] = weights.get("press_resistance", 0.3) * 1.2
        why += " Boosted further for a <b>must-win</b>, goal threat and press resistance take priority over balance."
    elif importance == "Rotation":
        why += " Rotation mode, freshest legs are favoured over the single best XI on paper."
    st.markdown(f'<div class="card"><b style="color:{ui.SKY}">Why this team:</b> {why}</div>', unsafe_allow_html=True)

    scored = compute_best_xi_score(squad, pr_ratings, {"press_resistance":weights["press_resistance"],"goal_contribution":weights["goal_threat"],"pass_quality":weights["passing"],"pressing_output":weights["pressing"]})
    scored = scored.merge(positions[["player","pos_group","position"]], on="player", how="left"); scored["pos_group"]=scored["pos_group"].fillna("MID")
    if importance == "Rotation":
        scored = scored.merge(availability[["player","load"]], on="player", how="left")
        scored["load"] = scored["load"].fillna(0)
        scored["xi_score"] = (scored["xi_score"] - scored["load"] * 0.25).round(1)
        scored = scored.sort_values("xi_score", ascending=False).reset_index(drop=True)
    rec_xi = recommend_xi(scored, positions, profile, key_player, formation=formation)

    ui.section("🔄 Your XI, swap any player: the pitch and balance update live")
    st.caption("Picking a player who's already in another slot swaps the two, so nobody can end up named twice.")
    chosen = []
    for grp in ["GK","DEF","MID","FWD"]:
        options = scored[scored["pos_group"]==grp]["player"].tolist()
        recs = rec_xi[rec_xi["pos_group"]==grp]["player"].tolist()
        st.markdown(f"<b style='color:{ui.SKY};font-family:Rajdhani'>{grp}</b>", unsafe_allow_html=True)
        cols = st.columns(formation[grp])
        chosen += position_picker(grp, formation[grp], options, recs,
        key_prefix=f"bxi_{opponent}_{formation_name}", cols=cols)

    cur = scored[scored["player"].isin(chosen)].merge(squad_norm[["player","n_press_res","n_goal","n_creativity","n_passing","n_workrate"]], on="player", how="left")
    rec = rec_xi.merge(squad_norm[["player","n_press_res","n_goal","n_creativity","n_passing","n_workrate"]], on="player", how="left")

    # interactive lineup pitch, built from the CURRENT selections, in slot order
    st.markdown(""); ui.section("🟢 Lineup")
    xi_rows = []
    for pl in chosen:
        r = cur[cur["player"]==pl]
        if not len(r): continue
        r = r.iloc[0]
        xi_rows.append({"player": pl, "pos_group": r["pos_group"], "ovr": player_overall(pl),
        "role": F.assign_role(r), "fam": F.role_familiarity(r)})
    st.plotly_chart(plotly_lineup(xi_rows, formation_name), use_container_width=True)
    st.caption("Tokens sit in their real positions for the chosen formation, coloured by overall rating. "
    "Hover for role + familiarity (＋/＋＋). Swap anyone above and the pitch updates instantly.")

    st.markdown(""); col1,col2 = st.columns(2)
    with col1:
        ui.section("📊 Team Balance")
        st.plotly_chart(radar_chart([F.team_radar(rec), F.team_radar(cur)], ["Recommended","Your pick"]), use_container_width=True)
    with col2:
        ui.section("⚖️ Trade-off")
        for ax,rv,cv in zip(F.RADAR_AXES, F.team_radar(rec), F.team_radar(cur)):
            d = cv-rv; color = ui.GREEN if d>0 else (ui.RED if d<0 else "#7E96C4"); arrow = "▲" if d>0 else ("▼" if d<0 else ", ")
            st.markdown(f"<div style='padding:6px 0;border-bottom:1px solid {ui.LINE}'>{ax}: <b style='color:{color}'>{cv:.0f} {arrow} {abs(d):.0f}</b></div>", unsafe_allow_html=True)

    st.markdown(""); ui.section("🎭 Roles & Familiarity")
    if "pos_group" in cur.columns:
        for grp in ["GK","DEF","MID","FWD"]:
            for _, r in cur[cur["pos_group"]==grp].iterrows():
                fam = F.role_familiarity(r)
                st.markdown(f"<div style='padding:5px 0;border-bottom:1px solid {ui.LINE}'><b>{r['player']}</b>, <span style='color:{ui.GOLD}'>{F.assign_role(r)}</span> <span style='color:{ui.GREEN};font-weight:700'>{fam}</span></div>", unsafe_allow_html=True)

    st.markdown(""); ui.section("🆚 Scenario Comparer: try a different style head-to-head")
    st.caption("Pick two tactical presets and see how the resulting XI's balance and projected result against "
    f"{opponent} differ, a quick way to pressure-test the auto recommendation before committing.")
    sc1, sc2 = st.columns(2)
    preset_names = list(F.TACTICAL_PRESETS.keys())
    pick_a = sc1.selectbox("Scenario A", preset_names, index=preset_names.index(preset_name), key="scA")
    pick_b = sc2.selectbox("Scenario B", preset_names,
    index=preset_names.index("Counter") if "Counter" in preset_names else min(1, len(preset_names)-1),
    key="scB")

    def _scenario_xi(name):
        w = dict(auto_weights) if F.TACTICAL_PRESETS[name] is None else dict(F.TACTICAL_PRESETS[name])
        sc = compute_best_xi_score(squad, pr_ratings, {"press_resistance": w["press_resistance"],
        "goal_contribution": w["goal_threat"], "pass_quality": w["passing"],
        "pressing_output": w["pressing"]})
        sc = sc.merge(positions[["player","pos_group","position"]], on="player", how="left")
        sc["pos_group"] = sc["pos_group"].fillna("MID")
        xi = recommend_xi(sc, positions, profile, key_player, formation=formation)
        return xi.merge(squad_norm[["player","n_press_res","n_goal","n_creativity","n_passing","n_workrate"]], on="player", how="left")

    xi_a, xi_b = _scenario_xi(pick_a), _scenario_xi(pick_b)
    radar_a, radar_b = F.team_radar(xi_a), F.team_radar(xi_b)
    st.plotly_chart(radar_chart([radar_a, radar_b], [pick_a, pick_b]), use_container_width=True)

    biggest_gap = max(zip(F.RADAR_AXES, radar_a, radar_b), key=lambda t: abs(t[1]-t[2]))
    winner = pick_a if biggest_gap[1] > biggest_gap[2] else pick_b
    st.markdown(ui.rec_box("info",
        f"The clearest split is on <b>{biggest_gap[0]}</b>, <b>{winner}</b> gives you more of it. "
        f"Pick whichever axis matters most against {opponent}'s {profile['press_level'].lower()}-press, "
        f"{profile['leaky']} defence."), unsafe_allow_html=True)


elif page == "Availability":
    ui.hero("Squad Availability & Load", "Who's fit, who's carrying a knock, who's overworked, who's a card away from a ban")
    ui.legend("What this shows", [
    "Before you pick a player, check here first: their status, how many minutes they've carried this "
    "season, whether they've come off injured recently, and whether a booking would cost you their "
    "next fixture.",
    "Status: 🟢 Available · 🩹 Knock (came off injured this season) · 🔵 High load (heavy recent minutes) · "
    "🟡 Booking risk (4+ yellows) · 🔴 Suspended (red card).",
    ])
    av = availability

    counts = av["status"].value_counts()
    c = st.columns(4)
    c[0].markdown(ui.stat_card("Fully available", int(counts.get("🟢 Available",0))), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Carrying a knock", int(counts.get("🩹 Knock",0))), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("High load", int(counts.get("🔵 High load",0))), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Booking/suspension", int(counts.get("🟡 Booking risk",0))+int(counts.get("🔴 Suspended",0))), unsafe_allow_html=True)

    # flags worth the manager's attention
    knocks = av[av["injury_subs"] > 0]
    highload = av[av["load"] >= 75]
    if len(knocks):
        st.markdown(ui.rec_box("warn", "Came off injured this season (monitor fitness): <b>" +
        ", ".join(short_name(p) for p in knocks["player"]) + "</b>."), unsafe_allow_html=True)
    if len(highload):
        st.markdown(ui.rec_box("info", "Heaviest workload, consider rotation: <b>" +
        ", ".join(short_name(p) for p in highload["player"].head(5)) + "</b>."), unsafe_allow_html=True)

    ui.section("Squad Status")
    show = av.copy()
    show["Player"] = show["player"].apply(short_name)
    show = show[["Player","status","apps","minutes","recent5_min","injury_subs","yellows","load"]]
    show.columns = ["Player","Status","Apps","Minutes","Last-5 mins","Injury subs","Yellows","Load"]
    show.index = range(1, len(show)+1)
    st.dataframe(show.style.background_gradient(subset=["Load"], cmap="RdYlGn_r"),
    use_container_width=True, height=560)

    ui.section("Workload (total minutes)")
    top = av.head(14).copy(); top["name"] = top["player"].apply(short_name)
    fig = px.bar(top, x="minutes", y="name", orientation="h", template="plotly_dark",
    color="load", color_continuous_scale=["#36E27A", ui.AMBER, ui.RED])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=440,
    yaxis=dict(autorange="reversed", title=""), xaxis=dict(title="Minutes played"),
    coloraxis_colorbar=dict(title="Load"), margin=dict(t=10,l=0,r=0,b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Load blends recent and season minutes into a single fatigue proxy, a heavier bar means less in the tank if you start him again straight away.")


elif page == "Pressing Game":
    ui.hero("Pressing Game", "How City presses, and who can handle being pressed")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Pressure Resistance"] + PAGE_TERMS["Pressing"])
    ui.legend("What this shows", [
    "Two sides of the same coin: how aggressively Manchester City presses as a team, and how well "
    "City's own players keep the ball when an opponent does the same to them.",
    "PPDA and pressure volume show the team's pressing intensity and where it happens on the pitch. "
    "The press resistance table and player map show who you can trust on the ball under pressure.",
    ])
    ui.section("City's Pressing")
    st.caption("PPDA counts how many passes the opponent strings together, on average, before City makes a tackle, "
    "interception or pressure. The lower it is, the less time anyone gets on the ball against City.")
    pe = F.pressing_effectiveness(city_events, OUR_TEAM)
    c = st.columns(3)
    c[0].markdown(ui.stat_card("PPDA", pe["ppda"], "lower = more aggressive"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Pressures / game", int(pe["pressures_pg"])), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("High-press / game", int(pe["high_press_pg"])), unsafe_allow_html=True)
    st.image(pitch_kde_img(pe["pressure_points"],"loc_x","loc_y","Blues"), use_container_width=True)
    st.caption("Brighter patches are where City presses most often this season, mostly the middle and final third if the press is working as intended.")
    pp = pe["per_player"].head(10).copy(); pp.columns = ["Player","Pressures","High Press","High %"]; pp.index = range(1,len(pp)+1)
    st.dataframe(pp, use_container_width=True, height=340)
    if pe["ppda"] <= 10:
        st.markdown(ui.rec_box("good", f"A PPDA of {pe['ppda']} is genuinely aggressive, City rarely lets opponents settle on the ball."), unsafe_allow_html=True)
    else:
        st.markdown(ui.rec_box("info", f"A PPDA of {pe['ppda']} is more measured than relentless, fine for controlling games but it gives a patient opponent time on the ball."), unsafe_allow_html=True)

    st.markdown(""); ui.section("Who Handles Being Pressed")
    st.caption("Pressured pass % is completion rate when an opponent is right on the player, normal % is completion rate "
    "when they aren't. PR Score blends the two into one 0-100 number, the higher it is, the safer that "
    "player is on the ball under a high press.")
    pr = pr_ratings
    show = pr[["player","pressured_pass_pct","unpressured_pass_pct","resistance_drop","pressured_attempts","PR_score"]].copy()
    show.columns = ["Player","Under Pressure %","Normal %","Drop","Attempts","PR Score"]; show.index = range(1,len(show)+1)
    st.dataframe(show.style.background_gradient(subset=["PR Score"], cmap="RdYlGn"), use_container_width=True, height=360)
    psel = st.selectbox("Player map", pr["player"].tolist())
    pz = pressure_resistance_zones(city_events, OUR_TEAM); pz = pz[pz["player"]==psel]
    fig, ax = plt.subplots(figsize=(9,6)); fig.patch.set_facecolor("#0E1B33")
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
    if len(pz):
        ok=pz[pz["success"]]; no=pz[~pz["success"]]
        pitch.scatter(ok["loc_x"],ok["loc_y"], ax=ax, s=70, color=ui.GREEN, alpha=.75, edgecolors="white", label="Retained", zorder=4)
        pitch.scatter(no["loc_x"],no["loc_y"], ax=ax, s=70, color=ui.RED, alpha=.75, edgecolors="white", label="Lost", zorder=4)
        ax.legend(facecolor="#101D3D", labelcolor="#E8EEF6", loc="upper left")
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption(f"Every pressured touch {short_name(psel)} has taken this season, green where he kept it, red where he lost it.")
    best, worst = pr.iloc[0], pr.iloc[-1]
    st.markdown(ui.rec_box("good", f"<b>{best['player']}</b> is your most press-resistant player (PR {best['PR_score']}), start him when you expect to see a high press."), unsafe_allow_html=True)
    if worst["PR_score"] < best["PR_score"] - 15:
        st.markdown(ui.rec_box("warn", f"<b>{worst['player']}</b> (PR {worst['PR_score']}) is the one most likely to give the ball straight back if pressed hard, keep him further from his own goal against a heavy press."), unsafe_allow_html=True)


elif page == "Set Pieces":
    ui.hero("Set Pieces", "Dead-ball delivery zones, and whether they're actually paying off")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Set Pieces"])
    ui.legend("What this shows", [
    "City's dead-ball output this season: how many goals come from set pieces, the split by type "
    "(corner, free kick, throw-in), and where deliveries into the box are typically aimed.",
    "The verdict at the bottom compares set-piece shots against open-play shots on conversion rate "
    "and average chance quality, the honest answer to whether the time spent drilling routines is worth it.",
    ])
    sp = set_piece_analysis(city_events, OUR_TEAM)
    c = st.columns(2)
    c[0].markdown(ui.stat_card("Goals from set pieces", sp["goals_from_set_pieces"]), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Total set pieces", sp["total_set_pieces"]), unsafe_allow_html=True)
    if len(sp["by_type"]):
        bt = sp["by_type"].reset_index(); bt.columns=["Type","Count"]
        fig = px.bar(bt, x="Type", y="Count", template="plotly_dark", color="Count", color_continuous_scale=["#101D3D",ui.SKY])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=300, margin=dict(t=10,l=0,r=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("How City's dead balls split by type this season, corners usually dominate the count.")
    dl = sp["deliveries"].dropna(subset=["end_x","end_y"])
    if len(dl):
        ui.section("Delivery Zones")
        st.image(vhalf_kde_img(dl, "end_x", "end_y", "Blues"), use_container_width=True)
        st.caption("Where City's dead-ball deliveries actually land inside the box, the brightest patch is the most rehearsed target area.")

    st.markdown(""); ui.section("Set Piece vs Open Play: Is It Working?")
    gi1, gi2 = st.columns(2)
    with gi1:
        st.markdown(f"<div class='card'><b style='color:{ui.SKY}'>Set pieces</b><br>"
        f"{sp['sp_shots']} shots · {sp['sp_goals']} goals · {sp['sp_conv_pct']}% conversion<br>"
        f"{sp['sp_xg_per_shot']} xG per shot</div>", unsafe_allow_html=True)
    with gi2:
        st.markdown(f"<div class='card'><b style='color:{ui.GOLD}'>Open play</b><br>"
        f"{sp['op_shots']} shots · {sp['op_goals']} goals · {sp['op_conv_pct']}% conversion<br>"
        f"{sp['op_xg_per_shot']} xG per shot</div>", unsafe_allow_html=True)
    if sp["sp_shots"] >= 5:
        if sp["sp_xg_per_shot"] >= sp["op_xg_per_shot"] and sp["sp_conv_pct"] >= sp["op_conv_pct"]:
            st.markdown(ui.rec_box("good", "Set pieces are generating better chances than open play this season, "
            "worth the training time and worth taking even a marginal one rather than recycling possession."), unsafe_allow_html=True)
        else:
            st.markdown(ui.rec_box("warn", f"Set pieces are producing a lower quality chance ({sp['sp_xg_per_shot']} xG/shot) than "
            f"open play ({sp['op_xg_per_shot']} xG/shot) this season. They're not a wasted source of shots, but they "
            "shouldn't be relied on as the main route to goal, keep working the ball into the box through play too."), unsafe_allow_html=True)
    else:
        st.caption("Not enough set-piece shots yet this season for a reliable comparison.")


        # ══════════════════════════════════════════════════════════════════════════════
        # LEAGUE
        # ══════════════════════════════════════════════════════════════════════════════
elif page == "Elo Strength":
    ui.hero("League Strength: Elo Ratings", "Self-adjusting Elo: P = Pⁿ⁻¹ + k·(w − wₑ)")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Elo Strength"])
    ui.legend("What this shows", [
    "A self-adjusting strength rating for every Premier League team this season, a win raises your "
    "rating, weighted by how surprising the result was (beating a stronger team moves it more).",
    "These ratings are what drives the win-probability numbers on Matchup and the Season Forecast simulation.",
    ])
    with st.spinner("Computing Elo..."):
        elo = load_elo()
    ce = elo[elo["team"]==OUR_TEAM]; oe = elo[elo["team"]==opponent]
    c = st.columns(3)
    if len(ce): c[0].markdown(ui.stat_card("Man City Elo", int(ce.iloc[0]["elo"]), f"Rank #{int(ce.iloc[0]['elo_rank'])}"), unsafe_allow_html=True)
    if len(oe): c[1].markdown(ui.stat_card(f"{opponent} Elo", int(oe.iloc[0]["elo"]), f"Rank #{int(oe.iloc[0]['elo_rank'])}"), unsafe_allow_html=True)
    if len(ce) and len(oe):
        dr = ce.iloc[0]["elo"]-oe.iloc[0]["elo"]; wp = round(1/(10**(-dr/400)+1)*100,1)
        c[2].markdown(ui.stat_card("Win probability", f"{wp}%", "from Elo gap"), unsafe_allow_html=True)
    ui.section("Full League Table")
    fig = px.bar(elo, x="team", y="elo", template="plotly_dark", color="elo", color_continuous_scale=[ui.RED, ui.AMBER, ui.SKY])
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False, height=420, xaxis_tickangle=-40, margin=dict(t=10,l=0,r=0,b=100))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Benchmark: a sound rating system beats chance by ~17% (Stefani, 1998).")


    # ══════════════════════════════════════════════════════════════════════════════
    # EXPECTED THREAT (xT)
    # ══════════════════════════════════════════════════════════════════════════════
elif page == "Expected Threat":
    ui.hero("Expected Threat (xT)", "Who actually moves us into dangerous areas, not just who passes")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Expected Threat"])
    ui.legend("What this shows", [
    "A pitch-zone value surface (brighter = more dangerous to have the ball there) used to credit "
    "players for moving the ball into threatening areas, not just for completing a pass.",
    "The leaderboards rank City's and the opponent's top ball-progressors by total threat added this season.",
    ])
    st.caption("Each pitch zone has a threat value = goals scored per touch from there. "
    "A pass or carry's xT = value(end) − value(start). We rank players by total xT added. "
    "Descriptive (data-derived value surface), no ML.")

    ui.section("Threat Surface")
    fig, ax = plt.subplots(figsize=(10,5.5)); fig.patch.set_facecolor("#0E1B33")
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
    ax.imshow(xt_grid, extent=[0,120,0,80], origin="lower", cmap="magma", alpha=.75, aspect="auto", zorder=1)
    ax.set_title("Threat value by zone (brighter = more dangerous)", color="#E8EEF6", fontsize=10)
    st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown(""); ui.section("Player Threat Map")
    ui.legend("What this shows", [
        "The same threat surface above, but broken down for one player: every zone is labelled with "
        "how much Expected Threat that player has personally added from passes and carries starting "
        "there this season. Brighter, higher-numbered cells are where his threat actually comes from.",
        "Goalkeepers are left out, since they barely feature in this kind of ball progression.",
    ])
    outfield = sorted(set(positions[positions["pos_group"] != "GK"]["player"]) & set(kpis["player"]))
    psel_xt = st.selectbox("Player", outfield,
                           index=outfield.index("Kevin De Bruyne") if "Kevin De Bruyne" in outfield else 0,
                           key="xt_player")
    pgrid = F.player_xt_zone_grid(city_events, OUR_TEAM, psel_xt, xt_grid, xt_gx, xt_gy)
    if pgrid.max() > 0:
        st.pyplot(xt_zone_heatmap(pgrid, xt_gx, xt_gy, short_name(psel_xt)), use_container_width=True); plt.close()
        st.caption("Attack direction: left to right. Each number is total Expected Threat added from that zone this season, not a single-touch average, so heavy-touch zones naturally accumulate more.")
        hot_i, hot_j = np.unravel_index(np.argmax(pgrid), pgrid.shape)
        hot_x = (xt_gx[hot_j] + xt_gx[hot_j+1]) / 2
        third = "defensive" if hot_x < 40 else ("middle" if hot_x < 80 else "final")
        st.markdown(ui.rec_box("good", f"<b>{short_name(psel_xt)}</b> generates most of his threat from the "
            f"{third} third, the brightest cell on the map. That's where his touches matter most this season."), unsafe_allow_html=True)
    else:
        st.info(f"Not enough located pass/carry data for {short_name(psel_xt)} to map this season.")

    st.markdown(""); ui.section("City's Top Ball-Progressors")
    xt = F.xt_added(city_events, OUR_TEAM, xt_grid, xt_gx, xt_gy).head(10)
    fig2 = px.bar(xt, x="xt_added", y="player", orientation="h", template="plotly_dark",
    color="xt_added", color_continuous_scale=["#101D3D", ui.CYAN, ui.SKY])
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
    height=400, yaxis=dict(autorange="reversed"), margin=dict(t=10,l=0,r=0,b=10))
    st.plotly_chart(fig2, use_container_width=True)
    if len(xt):
        st.markdown(ui.rec_box("good", f"<b>{xt.iloc[0]['player']}</b> adds the most threat through his passing and carrying, your main creative engine."), unsafe_allow_html=True)

    if need_scout or True:
        st.markdown(""); ui.section(f"{opponent}'s Top Progressors: Who To Limit")
        oxt = F.xt_added(load_opp_full(opponent), opponent, xt_grid, xt_gx, xt_gy).head(6)
        if len(oxt):
            st.dataframe(oxt.rename(columns={"player":"Player","xt_added":"xT Added"}).set_index("Player"), use_container_width=True)


            # ══════════════════════════════════════════════════════════════════════════════
            # MATCH REPORT (auto-generated, exportable)
            # ══════════════════════════════════════════════════════════════════════════════
            # ══════════════════════════════════════════════════════════════════════════════
            # MODEL VALIDATION (backtest)
            # ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Validation":
    ui.hero("Model Validation", "Does our Elo actually predict results? Tested against the season.")
    ui.legend("What this shows", [
    "An honest backtest of the Elo model: it walks the season chronologically, predicts each match "
    "using only the Elo ratings known <i>before</i> kickoff, then updates, no hindsight.",
    "Draws are excluded (Elo predicts a winner) and accuracy is checked against a naive "
    "always-pick-the-home-team baseline and a published benchmark, so you can judge if it's genuinely adding signal.",
    ])
    with st.spinner("Backtesting..."):
        allm = get_all_pl_matches()
        bt = F.backtest_elo(allm)
    c = st.columns(3)
    c[0].markdown(ui.stat_card("Elo accuracy", f"{bt['accuracy']}%", f"{bt['games']} decisive games"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Benchmark", f"~{bt['benchmark']}%", "Stefani (1998)"), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Home-win baseline", f"{bt['home_baseline']}%", "naive guess"), unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
    x=["Our Elo model","Stefani benchmark","Home-win baseline"],
    y=[bt["accuracy"], bt["benchmark"], bt["home_baseline"]],
    marker_color=[ui.SKY, ui.GOLD, "#7E96C4"], text=[f"{bt['accuracy']}%", f"~{bt['benchmark']}%", f"{bt['home_baseline']}%"],
    textposition="outside"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=380, yaxis=dict(title="Accuracy %", range=[0,80]), margin=dict(t=20,l=0,r=0,b=10))
    st.plotly_chart(fig, use_container_width=True)
    verdict = ("comfortably beats" if bt["accuracy"] > bt["home_baseline"]+3 else "edges")
    st.markdown(ui.rec_box("good",
    f"Our Elo model {verdict} the naive home-win baseline ({bt['accuracy']}% vs {bt['home_baseline']}%) "
    f"and lands near the ~{bt['benchmark']}% mark a sound rating system is expected to hit (Stefani, 1998). "
    f"This validates using it for the opponent-strength and win-probability features."), unsafe_allow_html=True)


    # ══════════════════════════════════════════════════════════════════════════════
    # PREDICT, SEASON FORECAST (Monte Carlo + prescriptive trends)
    # ══════════════════════════════════════════════════════════════════════════════
elif page == "Season Forecast":
    ui.hero("Season Forecast", "Where will City finish? A simulated outlook, not just past results")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Season Forecast"])
    ui.legend("What this shows", [
    "<b>Monte Carlo simulation</b>, we rate all 20 teams with Elo (the lecture rating system), turn each fixture into expected goals, and play out a full season hundreds of times.",
    "<b>Predicted finish</b>, the average final position across all simulations, plus the odds of the title, top four and relegation.",
    "<b>Form & projection</b>, pick a metric to see how it's trended and what it projects to over a full season (prescriptive, not just descriptive).",
    ])
    with st.spinner("Simulating the season 400×..."):
        elo = load_elo()
        mc = F.monte_carlo_season(elo, n_sims=400)
    c = st.columns(4)
    c[0].markdown(ui.stat_card("Predicted finish", f"{mc['avg_finish']:.0f}ᵗʰ", f"avg of {mc['n_sims']} sims"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Projected points", f"{mc['avg_points']:.0f}"), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Title odds", f"{mc['p_title']}%"), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Top-4 odds", f"{mc['p_top4']}%"), unsafe_allow_html=True)

    ui.section("Where We Might Finish")
    dist = mc["dist"].reset_index(); dist.columns = ["Position","Probability"]
    dist["Probability"] = (dist["Probability"]*100).round(1)
    figp = px.bar(dist, x="Position", y="Probability", template="plotly_dark",
    color="Probability", color_continuous_scale=["#101D3D", ui.SKY])
    figp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
    height=330, margin=dict(t=20,l=10,r=10,b=20), yaxis_title="Chance %", xaxis_title="Final position")
    st.plotly_chart(figp, use_container_width=True)
    most_likely = int(dist.loc[dist["Probability"].idxmax(), "Position"])
    st.caption(f"Each bar is the share of {mc['n_sims']} simulated seasons that ended with City in that exact position. "
    f"The tallest bar, {most_likely}{'st' if most_likely==1 else ('nd' if most_likely==2 else ('rd' if most_likely==3 else 'th'))} place, is the single most likely outcome, even if the average sits elsewhere.")
    st.markdown(ui.rec_box("good" if mc["p_top4"]>=50 else "info",
    f"Across {mc['n_sims']} simulated seasons City finish <b>{mc['avg_finish']:.0f}ᵗʰ on average</b> "
    f"(~{mc['avg_points']:.0f} pts), with a {mc['p_title']}% title chance and {mc['p_top4']}% top-four chance. "
    f"Best case {mc['best']}ᵗʰ, worst case {mc['worst']}ᵗʰ."), unsafe_allow_html=True)

    st.markdown(""); ui.section("Form & Projection")
    metric = st.selectbox("Metric", ["Points","Momentum","Goals For","Goals Against","Possession %","Passes","Shots"])
    df, proj = F.season_metric_series(city_events, city_matches, metric)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["match_week"], y=df["value"], name=metric, marker_color=ui.SKY, opacity=.55))
    fig.add_trace(go.Scatter(x=df["match_week"], y=df["rolling"], name="5-game trend", line=dict(color=ui.GOLD, width=3)))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    height=360, margin=dict(t=20,l=10,r=10,b=20),
    xaxis=dict(rangeslider=dict(visible=True), title="Matchweek"),
    yaxis=dict(title=metric), legend=dict(orientation="h", y=-.32))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Bars are {metric.lower()} match by match, the gold line is the 5-game rolling average. "
    "Read the gold line for the genuine trend; the bars on their own are too noisy game to game.")
    season_avg = df["value"].mean()
    trending = "trending up" if proj > season_avg * 1.05 else ("trending down" if proj < season_avg * 0.95 else "holding steady")
    if metric == "Points":
        st.markdown(ui.rec_box("info", f"On current points-per-game, City project to <b>{proj:.0f} points</b> over a full 38-game season, "
        f"and recent form is {trending} against the season average."), unsafe_allow_html=True)
    else:
        st.markdown(ui.rec_box("info", f"Recent 5-game average for {metric}: <b>{proj}</b> versus a season average of "
        f"{season_avg:.1f}, form is {trending}. Watch the gold trend line for momentum shifts."), unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════════════
        # PREDICT, LIVE MATCH ASSISTANT
        # ══════════════════════════════════════════════════════════════════════════════
elif page == "Live Assistant":
    ui.hero(f"Live Match Assistant: vs {opponent}", "Real-time prescriptive calls based on the game situation")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Live Assistant"])
    ui.legend("How this works", [
    "Set the minute and the scoreline state you're actually in, and the assistant turns the "
    f"{opponent} scouting profile (their press, their key man, whether they fade late, their danger "
    "men) into calls for that exact situation, not generic advice.",
    "Use it like a touchline earpiece: it tells you when to push, when to manage the game, who to "
    "mark tightest, and where the space is likely to be.",
    ])
    c = st.columns(2)
    minute = c[0].slider("Current minute", 1, 90, 70)
    state = c[1].radio("Score state", ["Leading","Level","Trailing"], horizontal=True)

    dm = load_danger(opponent)
    tips = F.match_assistant(profile, key_player, fatigue, dm, minute, state)
    ui.section(f"📣 Calls to make ({minute}', {state.lower()})")
    for tag, title, desc in tips:
        col = ui.GREEN if tag=="good" else (ui.AMBER if tag=="warn" else ui.SKY)
        icon = {"good":"✅","warn":"⚠️","info":"ℹ️"}[tag]
        st.markdown(f'<div class="card" style="border-left:5px solid {col}">'
        f'<div style="font-family:Rajdhani;font-weight:700;color:{col};font-size:1.05rem">{icon} {title}</div>'
        f'<div style="margin-top:3px">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown(""); ui.section("What's At Stake Right Now")
    gst = load_gamestate(opponent)
    their_press_now = gst.get(state, 0)
    wc1, wc2 = st.columns(2)
    with wc1:
        st.markdown(ui.stat_card(f"Their press while {state.lower()}", f"{their_press_now} pressures/min"), unsafe_allow_html=True)
    with wc2:
        kp = key_player["player"].split()[-1] if key_player else "n/a"
        st.markdown(ui.stat_card("Key man to track", kp), unsafe_allow_html=True)
    if their_press_now and max(gst.values() or [0]) > 0 and their_press_now == max(gst.values()):
        st.markdown(ui.rec_box("warn", f"This is their most intense pressing state, expect to be hurried on the ball right now, "
        "so the first pass after winning it back needs to be quick and simple."), unsafe_allow_html=True)
    elif their_press_now and their_press_now == min(gst.values()):
        st.markdown(ui.rec_box("good", "This is when they press least, the calmest moment in the game to work the ball patiently and probe for gaps."), unsafe_allow_html=True)

    st.markdown(""); ui.section("Suggested Substitution")
    sub_metric = {"Trailing": ("n_goal", "goal threat", "an extra attacking edge"),
    "Leading": ("n_workrate", "work-rate", "fresh legs to kill the tempo"),
    "Level": ("n_press_res", "ball retention under pressure", "control to tilt the midfield")}
    metric_col, metric_label, payoff = sub_metric[state]
    pool = squad_norm.sort_values(metric_col, ascending=False).reset_index(drop=True)
    sub = st.selectbox("Who do you actually want to bring on?", pool["player"].head(10).tolist(),
    help="Ranked by what this game state calls for, pick whoever's on your bench and the call below adjusts to them.")
    row = pool[pool["player"] == sub].iloc[0]
    rank = int(pool.index[pool["player"] == sub][0]) + 1
    tone = "good" if rank == 1 else ("info" if rank <= 3 else "warn")
    st.markdown(ui.rec_box(tone,
    f"<b>{sub}</b> ranks #{rank} of {len(pool)} in the squad for {metric_label} "
    f"({row[metric_col]:.0f}/100), bringing him on gives you {payoff}."), unsafe_allow_html=True)



    # ══════════════════════════════════════════════════════════════════════════════
    # INTELLIGENCE, TRAINED ML xG MODEL
    # ══════════════════════════════════════════════════════════════════════════════
elif page == "xG Model":
    ui.hero("Expected Goals: Trained Model", "A gradient-boosted xG model learned from shot geometry and spatial freeze-frames")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["xG Model"])
    ui.legend("What this shows", [
    "A real <b>machine-learning model</b> (gradient-boosted trees) trained to predict whether each shot becomes a goal.",
    "Features include distance & angle to goal, body part, and <b>spatial signals from freeze-frames</b>, defenders in the shooting lane and the keeper's distance from goal.",
    "Validated out-of-fold (no leakage). We report ROC-AUC, log-loss and a calibration curve, and benchmark against StatsBomb's own xG.",
    ])
    with st.spinner("Training xG model..."):
        xg = load_xg_model(city_events)
    m = xg["metrics"]

    c = st.columns(4)
    c[0].markdown(ui.stat_card("Shots trained on", f"{m['n_shots']}"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("ROC-AUC", f"{m['roc_auc']}", "discrimination"), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Log-loss", f"{m['log_loss']}", f"StatsBomb {m.get('sb_log_loss','N/A')}"), unsafe_allow_html=True)
    c[3].markdown(ui.stat_card("Brier", f"{m['brier']}", "calibration error"), unsafe_allow_html=True)

    cc = st.columns([1.1, 1])
    with cc[0]:
        ui.section("What drives a goal")
        imp = xg["importances"]
        fig = px.bar(imp.head(8), x="importance", y="feature", orientation="h",
        template="plotly_dark", color="importance",
        color_continuous_scale=["#101D3D", ui.CYAN, ui.SKY])
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, height=340,
        yaxis=dict(autorange="reversed"), margin=dict(t=10, l=0, r=0, b=10))
        st.plotly_chart(fig, use_container_width=True)
        top_feat = imp.iloc[0]["feature"].replace("_", " ")
        st.markdown(ui.rec_box("info", f"The model's strongest signal is <b>{top_feat}</b>, a spatial feature read from the freeze-frame, not just shot location."), unsafe_allow_html=True)
    with cc[1]:
        ui.section("Calibration")
        cal = xg["calibration"]
        figc = go.Figure()
        figc.add_trace(go.Scatter(x=[0, cal["predicted"].max()], y=[0, cal["predicted"].max()],
        mode="lines", line=dict(color="#5E76A6", dash="dash"), name="perfect"))
        figc.add_trace(go.Scatter(x=cal["predicted"], y=cal["observed"], mode="lines+markers",
        line=dict(color=ui.GOLD, width=3), marker=dict(size=8), name="model"))
        figc.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", height=340,
        xaxis=dict(title="Predicted", gridcolor="#1E2A40"),
        yaxis=dict(title="Observed", gridcolor="#1E2A40"),
        legend=dict(orientation="h", y=-.25), margin=dict(t=10, l=10, r=10, b=10))
        st.plotly_chart(figc, use_container_width=True)
        st.caption("Points on the diagonal = predicted goal probability matches the real goal frequency.")

    ui.section("Finishing vs the model, City")
    tbl = P.predict_xg_for_team(xg, OUR_TEAM).head(10)
    tbl = tbl.copy()
    tbl["player"] = tbl["player"].apply(short_name)
    show = tbl.rename(columns={"player": "Player", "shots": "Shots", "goals": "Goals",
    "model_xg": "Model xG", "overperformance": "+/− vs xG"})[
    ["Player", "Shots", "Goals", "Model xG", "+/− vs xG"]].set_index("Player")
    st.dataframe(show, use_container_width=True)
    if len(tbl):
        over = tbl.sort_values("overperformance", ascending=False).iloc[0]
        under = tbl.sort_values("overperformance").iloc[0]
        st.markdown(ui.rec_box("good", f"<b>{short_name(over['player'])}</b> is your most clinical finisher, {over['overperformance']:+.1f} goals above what the model expected from his chances."), unsafe_allow_html=True)
        if under["overperformance"] < -1:
            st.markdown(ui.rec_box("warn", f"<b>{short_name(under['player'])}</b> is underperforming his chances by {abs(under['overperformance']):.1f} goals, "
            "either a finishing slump worth working on in training, or a sign his shots are coming from positions the model rates as low quality."), unsafe_allow_html=True)


            # ══════════════════════════════════════════════════════════════════════════════
            # INTELLIGENCE, MATCH OUTCOME PREDICTOR (trained classifier)
            # ══════════════════════════════════════════════════════════════════════════════
elif page == "Match Predictor":
    ui.hero(f"Match Predictor: vs {opponent}", "A trained classifier turning City's form into Win / Draw / Loss probabilities")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Match Predictor"])
    ui.legend("What this shows", [
    "A <b>multinomial logistic regression</b> trained on City's season, predicting the result from pre-match features.",
    "Features: home/away, and rolling form (xG created, xG conceded, points) over recent games, strictly using games <i>before</i> each match, so there's no hindsight.",
    "Validated with cross-validation and compared against a majority-class baseline so you can see if it genuinely adds signal.",
    ])
    with st.spinner("Training outcome model..."):
        mm = load_match_model(city_events, city_matches)
    m = mm["metrics"]

    venue = st.radio("Venue for this fixture", ["Home", "Away"], horizontal=True)
    pred = P.predict_next_match(mm, venue == "Home")

    c = st.columns(3)
    c[0].markdown(ui.stat_card("Win", f"{pred.get('W',0)}%"), unsafe_allow_html=True)
    c[1].markdown(ui.stat_card("Draw", f"{pred.get('D',0)}%"), unsafe_allow_html=True)
    c[2].markdown(ui.stat_card("Loss", f"{pred.get('L',0)}%"), unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
    x=[pred.get("W", 0), pred.get("D", 0), pred.get("L", 0)],
    y=["Win", "Draw", "Loss"], orientation="h",
    marker_color=[ui.GREEN, ui.AMBER, ui.RED],
    text=[f"{pred.get('W',0)}%", f"{pred.get('D',0)}%", f"{pred.get('L',0)}%"],
    textposition="auto"))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)", height=240,
    yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 100], title="Probability %"),
    margin=dict(t=10, l=10, r=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    cc = st.columns(2)
    cc[0].markdown(ui.stat_card("CV accuracy", f"{int(m['cv_accuracy']*100)}%", f"{m['n_games']} games"), unsafe_allow_html=True)
    cc[1].markdown(ui.stat_card("Baseline", f"{int(m['baseline']*100)}%", "always-pick-majority"), unsafe_allow_html=True)

    ui.section("How form moves the prediction")
    st.caption("Model coefficients (standardised). Positive = pushes toward that result.")
    coef = mm["coefficients"].rename(columns={
    "is_home": "Home", "form_xg_for": "Form xG for",
    "form_xg_against": "Form xG against", "form_points": "Form points"})
    st.dataframe(coef, use_container_width=True)
    feat = mm["features"]
    if len(feat):
        last = feat.iloc[-1]
        st.markdown(ui.rec_box("info",
        f"Going into this fixture City's recent form is <b>{last['form_xg_for']:.1f}</b> xG created and "
        f"<b>{last['form_xg_against']:.1f}</b> conceded per game, on <b>{last['form_points']:.1f}</b> points/game. "
        f"That, plus the {venue.lower()} venue, drives the numbers above."), unsafe_allow_html=True)
    if m["cv_accuracy"] <= m["baseline"]:
        st.markdown(ui.rec_box("warn", "On a single 38-game season the model only matches the majority-class baseline, predictions are directionally useful but the sample is small. With more seasons the edge grows."), unsafe_allow_html=True)
    else:
        st.markdown(ui.rec_box("good", f"The model beats a majority-class guess by {int((m['cv_accuracy']-m['baseline'])*100)} points, "
        f"treat the {venue.lower()} win probability above as a genuine signal for team-talk confidence, not just a coin flip."), unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════════════
        # INTELLIGENCE, PLAYER FORM FORECAST (trained regressor)
        # ══════════════════════════════════════════════════════════════════════════════
elif page == "Form Forecast":
    ui.hero("Player Form Forecast", "A trained model projecting each player's next-match goal involvement")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Form Forecast"])
    ui.legend("What this shows", [
    "A <b>ridge regression</b> trained on every player-match in the season, predicting <b>goals + assists</b> in the next game.",
    "Features are a rolling 3-game window of xG, shots, key passes and goal involvement, so it reads <i>current</i> form, not season totals.",
    "Validated against a naive 'predict the average' baseline (lower error = the model is learning real form signal).",
    ])
    with st.spinner("Training form model..."):
        fm = load_form_model(city_events, city_matches)
    m = fm["metrics"]
    if fm.get("model") is None:
        st.markdown(ui.rec_box("warn", "Not enough player-match samples to train the form model on this dataset."), unsafe_allow_html=True)
    else:
        c = st.columns(3)
        c[0].markdown(ui.stat_card("Samples", f"{m['n']}", "player-matches"), unsafe_allow_html=True)
        c[1].markdown(ui.stat_card("Model error", f"{m['cv_mae']}", "MAE, involvements"), unsafe_allow_html=True)
        c[2].markdown(ui.stat_card("Naive error", f"{m['naive_mae']}", "predict-the-mean"), unsafe_allow_html=True)

        ui.section("Projected next-match goal involvement")
        proj = P.predict_player_form(fm, city_events, city_matches, top_n=12)
        proj = proj.copy(); proj["player"] = proj["player"].apply(short_name)
        fig = px.bar(proj, x="projected_involve", y="player", orientation="h",
        template="plotly_dark", color="projected_involve",
        color_continuous_scale=["#101D3D", ui.CYAN, ui.SKY],
        labels={"projected_involve": "Projected goals + assists", "player": ""})
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, height=440,
        yaxis=dict(autorange="reversed"), margin=dict(t=10, l=0, r=0, b=10))
        st.plotly_chart(fig, use_container_width=True)
        if len(proj):
            top = proj.iloc[0]
            st.markdown(ui.rec_box("good", f"<b>{top['player']}</b> is trending hottest, the model projects ~<b>{top['projected_involve']:.2f}</b> goal involvements next match from his recent form. Worth starting him if the team sheet is close."), unsafe_allow_html=True)
            cold = proj.iloc[-1]
            if cold["projected_involve"] < top["projected_involve"] * 0.25:
                st.markdown(ui.rec_box("info", f"<b>{cold['player']}</b> projects the lowest of this group right now, not a reason to drop him alone, but worth weighing against fresher options if you're picking between similar players."), unsafe_allow_html=True)
        if m["cv_mae"] < m["naive_mae"]:
            st.markdown(ui.rec_box("info", f"The model beats the naive baseline ({m['cv_mae']} vs {m['naive_mae']} mean error), recent form genuinely predicts the next game better than just assuming the season average."), unsafe_allow_html=True)


            # ══════════════════════════════════════════════════════════════════════════════
            # INTELLIGENCE, SPATIAL LAB (freeze-frame tracking analytics)
            # ══════════════════════════════════════════════════════════════════════════════
elif page == "Spatial Lab":
    ui.hero("Spatial Lab: Tracking Analytics", "Pitch control, defensive shape and shot pressure from freeze-frame positions")
    ui.glossary_box(GLOSSARY, PAGE_TERMS["Spatial Lab"])
    ui.legend("What this shows", [
    "At the instant of every shot, StatsBomb records the <b>exact position of every visible player</b>, the same positional signal a computer-vision tracker extracts from footage.",
    "From those positions we compute a <b>pitch-control surface</b> (who owns which space), defensive <b>compactness & line height</b>, and a <b>shot-pressure</b> read.",
    "This is the spatial foundation the xG model's freeze-frame features are built on.",
    ])

    tab1, tab2, tab3 = st.tabs(["⬛ Pitch Control", "🛡️ Defensive Shape", "🎯 Shot Pressure"])

    with tab1:
        ui.section("Space control at a key moment")
        st.caption("A representative City goal. Brighter blue = City controls that space, red = the opponent. "
        "Players are drawn at their tracked positions.")
        row, ff = SP.representative_frame(city_events, OUR_TEAM)
        if ff is not None and len(ff):
            XX, YY, ctrl = SP.pitch_control_grid(ff)
            fig, ax = plt.subplots(figsize=(10, 5.6)); fig.patch.set_facecolor("#0E1B33")
            pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
            ax.imshow(ctrl, extent=[0, 120, 0, 80], origin="lower", cmap="RdBu", alpha=.62,
            vmin=-1, vmax=1, aspect="auto", zorder=1)
            mates = ff[ff["teammate"]]; opps = ff[~ff["teammate"]]
            ax.scatter(mates["x"], mates["y"], s=120, c="#3DF5E4", edgecolors="white", zorder=3, label="City")
            ax.scatter(opps["x"], opps["y"], s=120, c="#FF5D6C", edgecolors="white", zorder=3, label="Opponent")
            ax.legend(loc="upper left", facecolor="#101D3D", edgecolor="#C7D2DE", labelcolor="#E8EEF6", fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()
            cs = SP.control_summary(ff)
            c = st.columns(2)
            c[0].markdown(ui.stat_card("City space control", f"{cs['attacking_control_pct']}%", "of the pitch"), unsafe_allow_html=True)
            c[1].markdown(ui.stat_card("Final-third control", f"{cs['final_third_control_pct']}%"), unsafe_allow_html=True)
            st.markdown(ui.rec_box("info", f"At the moment of this goal City controlled {cs['final_third_control_pct']}% of the final third, "
            "the kind of spatial dominance that turns half-chances into clear ones. Worth reviewing footage of similar moments in training."), unsafe_allow_html=True)
        else:
            st.info("No freeze-frame available for this selection.")

    with tab2:
        ui.section(f"{opponent}'s defensive block under fire")
        st.caption("Averaged across every shot they faced this season, measured from their tracked defenders' positions.")
        ds = SP.defensive_shape(opp_full, opponent, when_defending=True)
        if ds:
            c = st.columns(4)
            c[0].markdown(ui.stat_card("Line height", f"{ds['avg_line_height']}", "yds from own goal"), unsafe_allow_html=True)
            c[1].markdown(ui.stat_card("Block depth", f"{ds['avg_depth']}", "yds front-to-back"), unsafe_allow_html=True)
            c[2].markdown(ui.stat_card("Block width", f"{ds['avg_width']}", "yds side-to-side"), unsafe_allow_html=True)
            c[3].markdown(ui.stat_card("Compactness", f"{ds['avg_compactness']}", "lower = tighter"), unsafe_allow_html=True)
            deep_block = ds["avg_line_height"] < 25 and ds["avg_compactness"] < 16
            verdict = ("a deep, compact block, patience and width will be needed to pull them apart"
            if deep_block else "a higher line, there is space in behind to attack")
            st.markdown(ui.rec_box("info", f"{opponent} defend with {verdict}."), unsafe_allow_html=True)
            if deep_block:
                st.markdown(ui.rec_box("good", "Favour wide, creative passers over pure pace in behind here, weigh Build XI toward passing and creativity rather than straight-line speed."), unsafe_allow_html=True)
            else:
                st.markdown(ui.rec_box("good", "A higher defensive line rewards a fast outlet, weigh Build XI toward goal threat and pace in behind rather than patient build-up."), unsafe_allow_html=True)
        else:
            st.info("Not enough freeze-frame data for this opponent.")

    with tab3:
        ui.section("How crowded are City's shots?")
        pt = SP.shot_pressure_table(city_events, OUR_TEAM)
        ov = SP.pressure_overview(pt)
        if ov:
            c = st.columns(3)
            c[0].markdown(ui.stat_card("Defenders in lane", f"{ov['avg_defenders_in_cone']}", "avg per shot"), unsafe_allow_html=True)
            c[1].markdown(ui.stat_card("Clear sight", f"{ov['clear_sight_pct']}%", "of shots"), unsafe_allow_html=True)
            c[2].markdown(ui.stat_card("Nearest defender", f"{ov['avg_nearest_defender']}", "yds, avg"), unsafe_allow_html=True)
            comp = pd.DataFrame({
            "Situation": ["Clear sight", "Crowded (2+ in lane)"],
            "Conversion %": [ov["conv_clear"], ov["conv_crowded"]],
            "Shots": [ov["n_clear"], ov["n_crowded"]]})
            fig = px.bar(comp, x="Situation", y="Conversion %", template="plotly_dark",
            color="Situation", color_discrete_sequence=[ui.GREEN, ui.RED], text="Conversion %")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=320, margin=dict(t=10, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(ui.rec_box("good", f"City convert <b>{ov['conv_clear']}%</b> of clear-sight chances vs just <b>{ov['conv_crowded']}%</b> when the lane is crowded, creating cleaner looks is worth more than simply shooting more."), unsafe_allow_html=True)
        else:
            st.info("No shot freeze-frames available.")


            # ══════════════════════════════════════════════════════════════════════════════
            # INTELLIGENCE, VISION LAB (image → coordinates via computer vision)
            # ══════════════════════════════════════════════════════════════════════════════
elif page == "Vision Lab":
    ui.hero("Vision Lab: Tactical Snapshot", "Pause any moment, read the shape, make the call")
    ui.legend("What this shows", [
    "Two ways to read shape and movement straight from data: a <b>single-frame</b> computer-vision "
    "read of any tactical-cam image, and a <b>real possession replay</b> of a City goal, rebuilt "
    "from StatsBomb's own event locations and timings, not a simulation.",
    ])

    vtab1, vtab2 = st.tabs(["📷 Frame Detection", "🎬 Play Replay"])

    with vtab1:
        ui.legend("How a manager uses this", [
        "Upload a still from any match moment (or use the built-in frame). The app detects the players, works out the two teams, and maps them onto a top-down pitch.",
        "It then reads the picture like a coach would: <b>where you have or lack numbers</b> (overloads), <b>how compact your block is</b>, and <b>where the free space is</b>, each with a call to make.",
        "Under the hood it's a computer-vision pipeline (grass masking, line detection, blob detection, colour clustering). Works best on a clean, wide tactical-cam frame.",
        ])

        cu = st.columns([2,1])
        up = cu[0].file_uploader("Upload a pitch frame (optional)", type=["png", "jpg", "jpeg"])
        side = cu[1].radio("Which team is Man City?", ["Team A", "Team B"], help="Pick which detected team is yours so the advice is aimed correctly.")
        our_is_A = (side == "Team A")

        if up is not None:
            file_bytes = np.frombuffer(up.read(), np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                st.error("Couldn't read that image, try a different PNG/JPG."); st.stop()
            source = "your uploaded frame"
        else:
            img, _ = CV.synthetic_frame()
            source = "the built-in sample frame"

        with st.spinner("Reading the frame..."):
            res = CV.analyze_frame(img)
            m = res["metrics"]
            fdf = CV.players_to_frame_df(res)
            tr = CV.tactical_reads(fdf, our_team_is_A=our_is_A)

            # ---- MANAGER READ (primary output) ----
        ui.section("📋 The Manager's Read")
        if tr is None:
            st.info("Couldn't get a clear read, too few players detected. Try a cleaner, wider frame.")
        else:
            c = st.columns(4)
            c[0].markdown(ui.stat_card("Our players seen", tr["ours"]), unsafe_allow_html=True)
            c[1].markdown(ui.stat_card("Their players seen", tr["theirs"]), unsafe_allow_html=True)
            c[2].markdown(ui.stat_card("Block width", f"{tr['width']:.0f}m", tr["compact"]), unsafe_allow_html=True)
            c[3].markdown(ui.stat_card("Block depth", f"{tr['depth']:.0f}m"), unsafe_allow_html=True)
            for tag, txt in tr["recs"]:
                st.markdown(ui.rec_box(tag, txt), unsafe_allow_html=True)

                # overload heatmap (our numbers minus theirs, per zone)
            st.markdown(""); ui.section("Numbers By Zone (you − them)")
            grid = tr["grid"]
            fig0, ax0 = plt.subplots(figsize=(8, 4.6)); fig0.patch.set_facecolor("#0E1B33")
            pitch0 = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch0.draw(ax=ax0)
            ax0.imshow(grid, extent=[0,120,0,80], origin="lower", cmap="RdBu", alpha=.6, vmin=-3, vmax=3, aspect="auto")
            for r in range(3):
                for c2 in range(3):
                    v = int(grid[r, c2])
                    ax0.text(20+c2*40, 13.3+r*26.7, f"{v:+d}", color="#E8EEF6", ha="center", va="center",
                             fontsize=15, weight="bold")
            ax0.set_title("Blue = you have more players · Red = they do (you attack →)", color="#E8EEF6", fontsize=9)
            st.pyplot(fig0, use_container_width=True); plt.close()

            # ---- SUPPORTING EVIDENCE (detection + mapping) ----
        st.markdown(""); ui.section("🔎 Evidence: what the camera saw")
        st.caption(f"Analysing {source}. Detection: {m['n_players']} players · "
        f"{m['team0']}/{m['team1']} per team · {m['n_lines']} pitch lines · {m['pitch_coverage_pct']}% grass.")
        cc = st.columns(2)
        with cc[0]:
            ann = CV.annotate(img, res)
            st.image(ann, use_container_width=True, caption="Detected players (boxed by team) and pitch lines (cyan).")
        with cc[1]:
            fig, ax = plt.subplots(figsize=(7.5, 5)); fig.patch.set_facecolor("#0E1B33")
            pitch = Pitch(pitch_type="statsbomb", pitch_color="#0E1B33", line_color="#26406e"); pitch.draw(ax=ax)
            if len(fdf):
                ourm = fdf["teammate"] == our_is_A
                ax.scatter(fdf[ourm]["x"], fdf[ourm]["y"], s=140, c="#6CABDD", edgecolors="white", zorder=3, label="Man City")
                ax.scatter(fdf[~ourm]["x"], fdf[~ourm]["y"], s=140, c="#FF5D6C", edgecolors="white", zorder=3, label=opponent)
                if tr and tr["space_xy"]:
                    ax.scatter(*tr["space_xy"], s=420, marker="*", c="#F5D061", edgecolors="black", zorder=4, label="Free space")
                ax.legend(loc="upper left", facecolor="#101D3D", edgecolor="#C7D2DE", labelcolor="#E8EEF6", fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Classical CV is tuned for clean tactical-cam frames; crowds, ad boards and steep angles reduce accuracy, "
        "production trackers use trained neural networks instead.")

    with vtab2:
        ui.legend("How a manager uses this", [
        "Pick one of City's real goals this season. The pass/carry/shot trail, the distance of each "
        "step and the time it took are all <b>real StatsBomb event data</b>, there is no player-tracking "
        "video in this dataset, so we never invent positions between events.",
        "Distances are converted from StatsBomb's 120×80 pitch grid to an approximate real-world metre "
        "scale, treat them as indicative, not survey-accurate.",
        "At the shot itself, StatsBomb <b>does</b> sample every visible player's position (a freeze-frame) "
        ", that genuine snapshot is overlaid on the final frame of the replay.",
        ])
        moves = SP.city_goal_moves(city_events, OUR_TEAM)
        moves = moves.merge(city_matches[["match_id","match_week","opponent"]], on="match_id", how="left")
        if len(moves) == 0:
            st.info("No goals found to replay.")
        else:
            labels = [f"MW{int(r.match_week)} vs {r.opponent} · {int(r.minute)}:{int(r.second):02d}', {short_name(r.player)} ({r.play_pattern})"
                     for r in moves.itertuples()]
            pick = st.selectbox("Choose a goal to replay", range(len(moves)), format_func=lambda i: labels[i])
            row = moves.iloc[pick]
            steps, freeze_df = SP.build_possession_replay(city_events, OUR_TEAM, row["match_id"], row["possession"])

            if len(steps) < 2:
                st.info("This goal's possession has too few located events to replay.")
            else:
                total_m = round(sum(s["distance_m"] for s in steps), 1)
                speeds = [s["speed_kmh"] for s in steps if s["speed_kmh"]]
                elapsed = (steps[-1]["minute"]*60+steps[-1]["second"]) - (steps[0]["minute"]*60+steps[0]["second"])
                c = st.columns(4)
                c[0].markdown(ui.stat_card("Distance covered", f"{total_m}m"), unsafe_allow_html=True)
                c[1].markdown(ui.stat_card("Time elapsed", f"{elapsed}s"), unsafe_allow_html=True)
                c[2].markdown(ui.stat_card("Fastest segment", f"{max(speeds):.0f} km/h" if speeds else "N/A", "ball speed"), unsafe_allow_html=True)
                c[3].markdown(ui.stat_card("Actions", len(steps), "pass/carry/shot"), unsafe_allow_html=True)

                st.plotly_chart(replay_figure(steps, freeze_df), use_container_width=True)
                st.caption("Drag the slider or hit Play. The final frame overlays the real freeze-frame positions at the shot (sky = City, red = opponent).")

                quick = elapsed <= 8
                tempo = "a lightning-quick" if quick else "a patient, possession-based"
                st.markdown(ui.rec_box("good",
                    f"This was {tempo} move, {len(steps)} actions covering {total_m}m in {elapsed}s "
                    f"({row['play_pattern'].lower()}). "
                    + ("Quick transitions like this punish teams caught out of shape, worth drilling the same speed of thought."
                       if quick else
                       "Patient build-up like this works against deep blocks that can't be rushed, keep circulating until the gap opens.")
                    ), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELP, HOW TO USE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "How To Use":
    ui.hero("How To Use PitchInstinct", "Your manager's toolkit, explained in plain language")
    st.markdown("Pick your **opponent** and **formation** at the top, then work top to bottom: "
    "understand your squad, scout the opponent, pick the team, and look ahead. Here's what each section does.")
    st.markdown("**Finding your way around:** pages are grouped into dropdowns along the top bar, e.g. "
    "🏟️ Match Centre, click one to see and jump to its pages. Don't know where something lives? Use the "
    "🔍 search icon next to the formation picker to jump straight to any page by name. Most pages also have "
    "a small glossary box near the bottom that explains any jargon used on that page.")

    ui.section("🏟️ Match Centre, know your own team")
    for icon, n, d in [
            ("📊","Dashboard","Your snapshot before kickoff: season form, key stats, your best players, season headlines and the headline threat from the next opponent."),
            ("🃏","Squad","Every player as a card with data attributes, plus a quick best-XI view that reacts to the opponent you've picked. Filter by position."),
            ("👤","Player Profile","One player in depth: attributes, season numbers and a form trend across the games."),
            ("⚖️","Compare Players","Put two players side by side on a radar and head-to-head stats to settle selection calls."),
            ("🗓️","Season Trends","Points accumulation and rolling goals for/against across the season, plus where form has dipped and why."),
            ("⚡","Expected Threat","Who actually moves the ball into dangerous areas, not just who passes the most."),
            ]:
        st.markdown(ui.how_to_row(icon, n, d), unsafe_allow_html=True)

    ui.section("🎯 Opponent, scout who you're playing")
    for icon, n, d in [
            ("🔍","Scout Report","The full dossier: how they press, attack and defend, their danger men and likely XI."),
            ("⚔️","Matchup","What decides this game, your shape versus theirs, the key duels, and a projected scoreline with win odds."),
            ("🕸️","Who To Stop","Their creative hub, usually a midfielder, neutralise him to break their build-up."),
            ("🏃","Transition","Where space opens up the instant you win the ball back."),
            ("📉","Tactical Degradation","Whether their shape collapses late, and when to strike."),
            ]:
        st.markdown(ui.how_to_row(icon, n, d), unsafe_allow_html=True)

    ui.section("⚙️ Selection, pick the team")
    for icon, n, d in [
            ("👕","Build XI","Auto-picked XI to beat this opponent, with one-click tactical presets, live swaps, a balance radar and roles."),
            ("🩹","Availability","Squad fitness at a glance: knocks, suspensions and workload, built from real injury-sub, treatment and card events plus a minutes/load model."),
            ("🔥","Pressing Game","How hard and high your team presses, and which of your own players can be trusted on the ball when the opponent presses back."),
            ("🎯","Set Pieces","Your dead-ball threat and whether it's actually paying off compared to open play."),
            ]:
        st.markdown(ui.how_to_row(icon, n, d), unsafe_allow_html=True)

    ui.section("🔮 Predict, look ahead")
    for icon, n, d in [
            ("📈","Season Forecast","A simulated league finish (title, top four and relegation odds) plus form projections."),
            ("🎧","Live Assistant","In-game calls based on the minute and scoreline: when to push, when to manage the game, and who to bring on."),
            ]:
        st.markdown(ui.how_to_row(icon, n, d), unsafe_allow_html=True)

    ui.section("🧠 Intelligence, trained models, spatial and vision")
    for icon, n, d in [
            ("🎯","xG Model","A trained gradient-boosted Expected Goals model using shot geometry and spatial freeze-frame features, with full validation and a finishing leaderboard."),
            ("🔢","Match Predictor","A trained classifier turning City's form into Win, Draw or Loss probabilities for the next fixture."),
            ("📈","Form Forecast","A trained regressor projecting each player's next-match goals and assists from rolling form."),
            ("⬛","Spatial Lab","Tracking-style analytics from freeze-frame positions: pitch control, defensive shape and shot pressure."),
            ("📷","Vision Lab","A computer-vision frame reader, plus a real possession replay rebuilt from match event data."),
            ]:
        st.markdown(ui.how_to_row(icon, n, d), unsafe_allow_html=True)
