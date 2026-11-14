"""
PitchInstinct — UI system  ·  "Tactical Board" identity

Design thesis: this is a manager's match-day intelligence system, so the screen
should read like a floodlit tactical whiteboard and a broadcast stat strip — not
a generic SaaS dashboard.

Decisions (every one deliberate, not a default):
  • Palette  — floodlit-night. One structural accent (City sky), three accents
               each with EXACTLY one job: cyan = live/active/"now",
               gold = ratings/peaks, sky = structure; plus traffic-light
               semantics for good / caution / risk.
  • Type     — Oswald (condensed, stadium-signage feel) for display + all
               numerals, tight; Inter for body so the display carries
               personality. Numerals tabular so columns align like a scoreboard.
  • Signature— the "chalk line": a hairline rule with a short perpendicular tick,
               a literal pitch-marking, anchoring every section header.
  • Restraint— readouts are a single hairline-divided STAT STRIP (a broadcast
               lower-third), not a row of floating glowing cards.

Public API unchanged (hero, section, stat_card, rec_box, form_pills, attr_bar,
fut_card, legend, how_to_row) so pages need no edits.
"""

import streamlit as st

# ── palette ───────────────────────────────────────────────────────────────────
SKY    = "#6CABDD"
SKY_HI = "#8FC4ED"
CYAN   = "#5BE9D6"
GOLD   = "#F2C14E"
GREEN  = "#3FD77E"
AMBER  = "#FFC04D"
RED    = "#FF5D6C"

BASE   = "#060C18"
PANEL  = "#0E1B33"
PANEL2 = "#13233F"
INK    = "#0A1228"
NAVY   = "#0A1228"
NAVY2  = "#101D3D"
CARD   = "rgba(16,29,50,0.66)"
LINE   = "rgba(108,171,221,0.18)"
HAIR   = "rgba(143,196,237,0.16)"
TXT    = "#C9D8EE"
MUT    = "#7E93B5"


def inject_theme():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@300;400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    :root {{ --sky:{SKY}; --gold:{GOLD}; --cyan:{CYAN}; --green:{GREEN}; --red:{RED}; --amber:{AMBER};
        /* liquid-glass material tokens */
        --glass-fill: linear-gradient(135deg, rgba(120,168,224,0.12), rgba(120,168,224,0.04) 46%, rgba(8,16,32,0.10));
        --glass-blur: blur(22px) saturate(165%);
        --glass-shadow: 0 18px 44px -22px rgba(0,0,0,.78);
        --glass-inset: inset 0 1px 0 rgba(255,255,255,.22), inset 0 -1px 0 rgba(255,255,255,.05);
    }}

    /* ============================ base ============================ */
    [data-testid="stAppViewContainer"]{{
        background:
          radial-gradient(1100px 520px at 84% -12%, rgba(108,171,221,0.10), transparent 62%),
          radial-gradient(820px 460px at -8% 112%, rgba(91,233,214,0.06), transparent 58%),
          linear-gradient(168deg, #05090F 0%, #081226 52%, #060C18 100%);
    }}
    /* drifting light orbs — what the frosted glass refracts over */
    [data-testid="stAppViewContainer"]::before{{
        content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
        background:
          radial-gradient(460px circle at 14% 22%, rgba(108,171,221,0.22), transparent 60%),
          radial-gradient(420px circle at 86% 68%, rgba(91,233,214,0.15), transparent 60%),
          radial-gradient(360px circle at 64% 14%, rgba(242,193,78,0.10), transparent 56%),
          radial-gradient(400px circle at 38% 88%, rgba(108,171,221,0.12), transparent 60%);
        animation:drift 26s ease-in-out infinite alternate;
    }}
    @keyframes drift {{
        0%   {{ transform:translate3d(0,0,0) scale(1); }}
        50%  {{ transform:translate3d(2.5%, -2%, 0) scale(1.06); }}
        100% {{ transform:translate3d(-2%, 2.5%, 0) scale(1.02); }}
    }}
    [data-testid="stHeader"]{{ background:transparent; }}
    .block-container{{ padding-top:2.0rem; max-width:1280px; position:relative; z-index:1; }}

    /* ---- the liquid-glass material (lit rim + specular highlight) ---- */
    .glass{{
        position:relative; border-radius:18px;
        background:var(--glass-fill);
        -webkit-backdrop-filter:var(--glass-blur); backdrop-filter:var(--glass-blur);
        box-shadow:var(--glass-shadow), var(--glass-inset);
        border:1px solid rgba(255,255,255,.06);
    }}
    /* lit rim: a 1px gradient border, bright where light hits (top-left) */
    .glass::before{{
        content:""; position:absolute; inset:0; border-radius:inherit; padding:1px; pointer-events:none;
        background:linear-gradient(135deg, rgba(255,255,255,.55), rgba(180,214,247,.12) 38%, transparent 62%);
        -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite:xor; mask-composite:exclude;
    }}
    /* specular: soft pool of light on the upper surface */
    .glass::after{{
        content:""; position:absolute; top:0; left:8%; right:34%; height:46%; border-radius:inherit;
        background:radial-gradient(120% 90% at 30% 0%, rgba(255,255,255,.16), transparent 72%);
        pointer-events:none;
    }}

    html, body, [class*="css"]{{ font-family:'Inter',system-ui,sans-serif; color:{TXT}; }}
    p, span, label, li, .stMarkdown{{ color:{TXT}; }}
    h1,h2,h3,h4{{ font-family:'Oswald',sans-serif !important; color:#EAF3FF !important;
        letter-spacing:.4px; font-weight:600; }}
    .os{{ font-family:'Oswald',sans-serif; font-feature-settings:'tnum' 1; }}

    @keyframes rise {{ from{{opacity:0; transform:translateY(10px);}} to{{opacity:1; transform:none;}} }}
    @keyframes blink {{ 0%,100%{{opacity:1;}} 50%{{opacity:.35;}} }}
    .block-container > div{{ animation:rise .42s cubic-bezier(.2,.7,.3,1) both; }}
    @media (prefers-reduced-motion: reduce){{
        .block-container > div{{ animation:none; }}
        .live-dot{{ animation:none !important; }}
        [data-testid="stAppViewContainer"]::before{{ animation:none !important; }}
    }}

    /* ============================ sidebar ============================ */
    [data-testid="stSidebar"]{{
        background:linear-gradient(180deg,#05090F 0%, #0A1730 100%);
        border-right:1px solid {HAIR};
    }}
    [data-testid="stSidebar"] .block-container{{ padding-top:1.2rem; }}
    .brand{{ font-family:'Oswald',sans-serif; font-weight:700; font-size:1.24rem; color:#fff;
        letter-spacing:.3px; text-transform:uppercase; line-height:1.05; white-space:nowrap; }}
    .brand span{{ color:{SKY}; }}
    .navlabel{{ font-family:'Oswald',sans-serif; font-size:.72rem; letter-spacing:2.5px;
        text-transform:uppercase; color:{MUT}; margin:10px 0 4px; }}
    [data-testid="stSidebar"] [data-testid="stExpander"]{{
        border:none !important; background:transparent !important; }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary{{
        font-family:'Oswald',sans-serif; letter-spacing:1px; text-transform:uppercase;
        font-size:.86rem; color:#C7D7EF; padding:6px 4px; border-radius:8px; }}
    [data-testid="stSidebar"] [data-testid="stExpander"] summary:hover{{ color:#fff; }}
    [data-testid="stSidebar"] .stButton button{{
        background:rgba(15,27,51,0); border:1px solid transparent; border-radius:9px;
        color:#AFC2E0; font-family:'Inter'; font-weight:500; font-size:.9rem;
        text-align:left; justify-content:flex-start; padding:6px 12px;
        transition:background .14s ease, border-color .14s ease, color .14s ease; }}
    [data-testid="stSidebar"] .stButton button:hover{{
        background:rgba(108,171,221,.10); border-color:{HAIR}; color:#EAF3FF; }}

    /* ============================ hero ============================ */
    .hero{{
        position:relative; border-radius:16px; padding:24px 28px 22px; margin-bottom:22px;
        background:linear-gradient(180deg, rgba(20,38,68,.92), rgba(10,20,40,.96));
        border:1px solid {LINE}; overflow:hidden;
        box-shadow:0 20px 46px -26px rgba(0,0,0,.85), inset 0 1px 0 rgba(143,196,237,.10);
    }}
    .hero::before{{ content:""; position:absolute; inset:0; pointer-events:none; opacity:.5;
        background:
          radial-gradient(circle at 88% 50%, transparent 53px, rgba(143,196,237,.14) 54px, transparent 55px),
          linear-gradient(90deg, transparent calc(88% - .5px), rgba(143,196,237,.14) 88%, transparent calc(88% + .5px));
    }}
    .hero .eyebrow{{ display:flex; align-items:center; gap:10px; margin-bottom:8px; position:relative; }}
    .hero .eyebrow .tick{{ width:26px; height:2px; background:{SKY}; box-shadow:0 0 10px rgba(108,171,221,.6); }}
    .hero .eyebrow span{{ font-family:'Oswald'; font-size:.72rem; letter-spacing:3px;
        text-transform:uppercase; color:{SKY_HI}; }}
    .hero h1{{ margin:0; font-size:2.05rem; font-weight:700; color:#fff !important;
        text-transform:uppercase; letter-spacing:.6px; line-height:1.02; font-family:'Oswald'; position:relative; }}
    .hero p{{ margin:7px 0 0; color:#B9CCEA; font-size:.98rem; max-width:70ch; position:relative; }}

    /* ============================ chalk section header ============================ */
    .sec{{ display:flex; align-items:baseline; gap:12px; margin:26px 0 14px; position:relative; }}
    .sec h3{{ margin:0; font-size:1.04rem; text-transform:uppercase; letter-spacing:2px;
        color:#EAF3FF !important; white-space:nowrap; font-weight:600; }}
    .sec .rule{{ flex:1; height:1px; background:linear-gradient(90deg, {SKY} 0, {HAIR} 38px, {HAIR} 100%); position:relative; }}
    .sec .rule::before{{ content:""; position:absolute; left:0; top:-4px; width:1px; height:9px; background:{SKY}; }}

    /* ============================ STAT STRIP ============================ */
    div[data-testid="stHorizontalBlock"]:has(> div [data-sc]) {{
        background:linear-gradient(180deg, rgba(17,30,55,.72), rgba(11,21,40,.72));
        border:1px solid {LINE}; border-radius:14px; padding:4px 2px; gap:0 !important;
        box-shadow:inset 0 1px 0 rgba(143,196,237,.07), 0 16px 34px -28px rgba(0,0,0,.8);
    }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-sc]) > div[data-testid="stColumn"]{{
        border-right:1px solid {HAIR}; }}
    div[data-testid="stHorizontalBlock"]:has(> div [data-sc]) > div[data-testid="stColumn"]:last-child{{
        border-right:none; }}
    .sc{{ text-align:center; padding:14px 10px 13px; }}
    .sc .v{{ font-family:'Oswald'; font-feature-settings:'tnum' 1; font-size:2.05rem;
        font-weight:600; color:#EAF3FF; line-height:1; letter-spacing:.5px; }}
    .sc .l{{ font-family:'Oswald'; font-size:.68rem; letter-spacing:2px; text-transform:uppercase;
        color:{MUT}; margin-top:8px; font-weight:500; }}
    .sc .s{{ font-size:.7rem; color:{SKY}; margin-top:4px; font-family:'Inter'; }}
    .sc.good .v{{ color:{GREEN}; }} .sc.warn .v{{ color:{AMBER}; }}
    .sc.risk .v{{ color:{RED}; }}  .sc.peak .v{{ color:{GOLD}; }} .sc.live .v{{ color:{CYAN}; }}

    .tile{{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:15px 12px;
        text-align:center; }}
    .tile .v{{ font-family:'Oswald'; font-feature-settings:'tnum' 1; font-size:2rem; font-weight:600;
        color:#EAF3FF; line-height:1; }}
    .tile .l{{ font-family:'Oswald'; font-size:.68rem; letter-spacing:2px; text-transform:uppercase; color:{MUT}; margin-top:7px; }}

    /* ============================ cards ============================ */
    .card{{ background:{CARD}; border:1px solid {LINE}; border-radius:14px; padding:16px 18px;
        margin-bottom:12px; transition:border-color .16s ease, transform .16s ease; }}
    .card:hover{{ border-color:rgba(108,171,221,.34); transform:translateY(-2px); }}

    /* ============================ readout (rec_box) ============================ */
    .rec{{ position:relative; border-radius:10px; padding:12px 16px 12px 18px; margin:9px 0;
        background:rgba(14,27,51,.62); border:1px solid {HAIR};
        border-left:3px solid {SKY}; font-size:.95rem; line-height:1.55; color:#D6E4F6; }}
    .rec .kick{{ font-family:'Oswald'; font-size:.64rem; letter-spacing:2px; text-transform:uppercase;
        display:block; margin-bottom:3px; color:{SKY}; }}
    .rec b{{ color:{SKY_HI}; font-weight:600; }}
    .rec-good{{ border-left-color:{GREEN}; }} .rec-good .kick{{ color:{GREEN}; }}
    .rec-warn{{ border-left-color:{AMBER}; }} .rec-warn .kick{{ color:{AMBER}; }}
    .rec-info{{ border-left-color:{SKY}; }}

    /* ============================ pills (form) ============================ */
    .pill{{ display:inline-flex; align-items:center; justify-content:center; min-width:24px;
        height:24px; padding:0 8px; border-radius:6px; font-size:.78rem; font-weight:600;
        margin:2px; font-family:'Oswald'; letter-spacing:.5px; }}
    .pw{{ background:rgba(63,215,126,.15); color:{GREEN}; border:1px solid rgba(63,215,126,.45); }}
    .pd{{ background:rgba(255,192,77,.14); color:{AMBER}; border:1px solid rgba(255,192,77,.42); }}
    .pl{{ background:rgba(255,93,108,.14); color:{RED};   border:1px solid rgba(255,93,108,.42); }}

    .live-dot{{ display:inline-block; width:8px; height:8px; border-radius:50%; background:{CYAN};
        box-shadow:0 0 10px {CYAN}; margin-right:7px; animation:blink 1.6s infinite; }}

    /* ============================ FUT-style player card ============================ */
    .fut{{ width:100%; border-radius:16px; padding:15px 16px 17px; position:relative;
        overflow:hidden; border:1px solid rgba(255,255,255,.16);
        box-shadow:0 18px 38px -22px rgba(0,0,0,.85); transition:transform .18s ease; }}
    .fut:hover{{ transform:translateY(-4px); }}
    .fut::after{{ content:""; position:absolute; inset:0; pointer-events:none;
        background:linear-gradient(180deg, rgba(255,255,255,.10), transparent 30%); }}
    .fut-gold{{ background:linear-gradient(157deg,#2c2408,#6a521166 40%,#caa544); }}
    .fut-blue{{ background:linear-gradient(157deg,#0a1d36,#15396f 42%,#3f86c9); }}
    .fut-silver{{ background:linear-gradient(157deg,#222a36,#4a586c 42%,#8fa2b8); }}
    .fut-top{{ display:flex; gap:12px; align-items:flex-start; justify-content:space-between; }}
    .fut-ovr{{ font-family:'Oswald'; font-feature-settings:'tnum' 1; font-size:2.5rem; font-weight:700;
        line-height:.86; color:#fff; }}
    .fut-pos{{ font-family:'Oswald'; font-size:.92rem; font-weight:500; color:rgba(255,255,255,.88);
        border-top:1px solid rgba(255,255,255,.45); margin-top:3px; padding-top:2px; text-align:center;
        letter-spacing:1px; }}
    .fut-photo{{ width:62px; height:62px; border-radius:50%; object-fit:cover; flex-shrink:0;
        border:2px solid rgba(255,255,255,.55); box-shadow:0 4px 12px -3px rgba(0,0,0,.6);
        background:rgba(0,0,0,.18); }}
    .fut-name{{ font-family:'Oswald'; font-size:1.22rem; font-weight:600; color:#fff;
        text-transform:uppercase; text-align:center; margin:9px 0 5px; letter-spacing:.5px;
        border-top:1px solid rgba(255,255,255,.32); padding-top:8px; }}
    .fut-attrs{{ display:grid; grid-template-columns:1fr 1fr; gap:3px 16px; margin-top:6px; }}
    .fut-attr{{ display:flex; justify-content:space-between; font-family:'Oswald';
        font-feature-settings:'tnum' 1; font-weight:500; color:#fff; font-size:.95rem; letter-spacing:.5px; }}
    .fut-attr span:first-child{{ color:rgba(255,255,255,.78); }}

    /* ============================ attribute bar ============================ */
    .abar-wrap{{ margin:8px 0; }}
    .abar-top{{ display:flex; justify-content:space-between; font-size:.8rem; color:#BACBE6;
        margin-bottom:4px; font-family:'Inter'; }}
    .abar-top .n{{ font-family:'Oswald'; font-feature-settings:'tnum' 1; font-weight:600; }}
    .abar-track{{ height:7px; border-radius:5px; background:rgba(255,255,255,.07); overflow:hidden; }}
    .abar-fill{{ height:100%; border-radius:5px; }}

    /* ============================ Streamlit widgets ============================ */
    .stRadio [role="radiogroup"]{{ gap:6px; }}
    .stRadio [role="radiogroup"] label{{ background:rgba(14,27,51,.55); border:1px solid {HAIR};
        border-radius:9px; padding:7px 13px; transition:all .14s ease; font-family:'Inter'; }}
    .stRadio [role="radiogroup"] label:hover{{ border-color:{SKY}; background:rgba(108,171,221,.10); }}
    [data-baseweb="select"] > div{{ background:rgba(14,27,51,.7) !important; border-color:{HAIR} !important;
        border-radius:9px !important; font-family:'Inter'; }}
    .stTabs [data-baseweb="tab-list"]{{ gap:4px; border-bottom:1px solid {HAIR}; }}
    .stTabs [data-baseweb="tab"]{{ background:transparent; border-radius:8px 8px 0 0; padding:8px 16px;
        font-family:'Oswald'; font-weight:500; letter-spacing:1px; text-transform:uppercase;
        font-size:.82rem; color:{MUT}; }}
    .stTabs [data-baseweb="tab"]:hover{{ color:#C7D7EF; }}
    .stTabs [aria-selected="true"]{{ background:linear-gradient(180deg, rgba(108,171,221,.22), rgba(108,171,221,.06));
        color:#EAF3FF !important; box-shadow:inset 0 -2px 0 {SKY}; }}
    [data-testid="stMetricValue"]{{ color:#EAF3FF !important; font-family:'Oswald'; font-feature-settings:'tnum' 1; }}
    [data-testid="stMetricLabel"]{{ font-family:'Oswald'; letter-spacing:1.5px; text-transform:uppercase; color:{MUT}; }}
    .stSlider [data-baseweb="slider"] [role="slider"]{{ background:{SKY}; }}
    div[data-testid="stDataFrame"]{{ border:1px solid {HAIR}; border-radius:12px; }}
    [data-testid="stExpander"]{{ border:1px solid {HAIR}; border-radius:12px; }}
    hr{{ border-color:{HAIR}; }}
    ::selection{{ background:rgba(108,171,221,.32); }}

    button:focus-visible, [role="button"]:focus-visible, a:focus-visible{{
        outline:2px solid {SKY}; outline-offset:2px; }}
    </style>
    """, unsafe_allow_html=True)


# ── components (API-compatible with the previous ui.py) ───────────────────────
def hero(title, subtitle, eyebrow="Match Intelligence"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow"><span class="tick"></span>'
        f'<span>{eyebrow}</span></div><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True)


def section(title):
    st.markdown(f'<div class="sec"><h3>{title}</h3><div class="rule"></div></div>',
                unsafe_allow_html=True)


_SC_TONES = {"good", "warn", "risk", "peak", "live"}

def stat_card(label, value, sub="", tone=""):
    t = f" {tone}" if tone in _SC_TONES else ""
    s = f'<div class="s">{sub}</div>' if sub else ""
    return (f'<div data-sc class="sc{t}"><div class="v os">{value}</div>'
            f'<div class="l">{label}</div>{s}</div>')


def rec_box(tag, text, kicker=None):
    default_kick = {"good": "Edge", "warn": "Watch", "info": "Read"}.get(tag, "Note")
    k = kicker if kicker is not None else default_kick
    return f'<div class="rec rec-{tag}"><span class="kick">{k}</span>{text}</div>'


def form_pills(results):
    html = ""
    for r in results:
        cls = {"W": "pw", "D": "pd", "L": "pl"}.get(r, "pd")
        html += f'<span class="pill {cls}">{r}</span>'
    return html


def _attr_color(v):
    if v >= 80: return GOLD
    if v >= 68: return SKY
    if v >= 52: return AMBER
    return RED

def attr_bar(label, value):
    c = _attr_color(value)
    return (f'<div class="abar-wrap"><div class="abar-top"><span>{label}</span>'
            f'<span class="n" style="color:{c}">{int(value)}</span></div>'
            f'<div class="abar-track"><div class="abar-fill" '
            f'style="width:{min(value,99)}%;background:linear-gradient(90deg,{c}99,{c})"></div></div></div>')


def fut_card(name, overall, position, attrs, image=None):
    tier = "fut-gold" if overall >= 82 else ("fut-blue" if overall >= 72 else "fut-silver")
    short = name.split()[-1] if name else ""
    rows = "".join(f'<div class="fut-attr"><span>{k}</span><span>{int(v)}</span></div>'
                   for k, v in attrs.items())
    photo = f'<img class="fut-photo" src="{image}" alt="{short}">' if image else ""
    return (f'<div class="fut {tier}">'
            f'<div class="fut-top"><div><div class="fut-ovr os">{overall}</div>'
            f'<div class="fut-pos">{position}</div></div>{photo}</div>'
            f'<div class="fut-name">{short}</div>'
            f'<div class="fut-attrs">{rows}</div></div>')


def legend(title, items):
    bullets = "".join(f'<li style="margin:4px 0">{it}</li>' for it in items)
    return st.markdown(
        f'<div class="card" style="background:rgba(12,24,46,.5)">'
        f'<div style="font-family:Oswald;font-weight:600;color:{SKY};letter-spacing:2px;'
        f'text-transform:uppercase;font-size:.72rem;margin-bottom:7px">{title}</div>'
        f'<ul style="margin:0 0 0 18px;padding:0;color:#BBCCE8;font-size:.9rem;line-height:1.5">{bullets}</ul></div>',
        unsafe_allow_html=True)


def how_to_row(icon, name, desc):
    return (f'<div class="card" style="display:flex;gap:14px;align-items:flex-start">'
            f'<div style="font-size:1.4rem;line-height:1">{icon}</div>'
            f'<div><div style="font-family:Oswald;font-weight:600;color:{SKY_HI};font-size:1.02rem;'
            f'letter-spacing:.5px">{name}</div>'
            f'<div style="color:#BBCCE8;font-size:.92rem;margin-top:2px">{desc}</div></div></div>')


def live_badge(text="Live"):
    return (f'<span class="os" style="color:{CYAN};font-size:.74rem;letter-spacing:2px;'
            f'text-transform:uppercase"><span class="live-dot"></span>{text}</span>')
