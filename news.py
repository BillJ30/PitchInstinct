"""
PitchInstinct: Season News
A short, honestly-curated set of real, dated Premier League 2015/16 storylines, plus a
season snapshot computed live from real match results. Nothing here is invented: the
curated list is real history with real dates, and is only ever shown up to the date of
the fixture the manager is preparing for, so the app never spoils what's still ahead that
season. The snapshot (league position, in-form sides) is built purely from match scorelines
already in the cached data, also cut off at that same date.
"""

import pandas as pd

CURATED_STORYLINES = [
    ("2015-10-09", "Jurgen Klopp is appointed Liverpool manager, replacing Brendan Rodgers."),
    ("2015-10-26", "Aston Villa are still searching for a way out of the bottom three after sacking Tim Sherwood."),
    ("2015-12-19", "Claudio Ranieri's Leicester City go top of the table at Christmas, defying the bookies' relegation odds."),
    ("2016-02-01", "Pep Guardiola is confirmed as Manchester City's next manager, arriving for next season."),
    ("2016-03-14", "Leicester City's title challenge refuses to fade as the run-in approaches."),
    ("2016-05-02", "Leicester City are crowned Premier League champions in one of English football's great upsets."),
]


def curated_news(cutoff_date, max_items=3):
    """Real, dated storylines that have actually happened by cutoff_date.
    Returns (kind, text) pairs so the caller can tag them as history."""
    cutoff = str(cutoff_date)
    items = [(d, t) for d, t in CURATED_STORYLINES if d <= cutoff]
    items.sort(key=lambda x: x[0], reverse=True)
    return [("history", t) for _, t in items[:max_items]]


def _ordinal(n):
    if 10 <= n % 100 <= 20:
        return f"{n}th"
    suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def season_snapshot(all_matches, cutoff_date):
    """A mini league table plus each team's last-5 points, built only from matches
    played strictly before cutoff_date so the picture reflects what's known so far."""
    m = all_matches[all_matches["match_date"] < str(cutoff_date)].copy()
    teams = pd.unique(m[["home_team", "away_team"]].values.ravel())
    rows, form_pts = [], {}
    for t in teams:
        home = m[m["home_team"] == t]; away = m[m["away_team"] == t]
        played = len(home) + len(away)
        if played == 0:
            continue
        gf = int(home["home_score"].sum() + away["away_score"].sum())
        ga = int(home["away_score"].sum() + away["home_score"].sum())
        pts = int((home["home_score"] > home["away_score"]).sum() * 3
                  + (away["away_score"] > away["home_score"]).sum() * 3
                  + (home["home_score"] == home["away_score"]).sum()
                  + (away["away_score"] == away["home_score"]).sum())
        rows.append({"team": t, "played": played, "pts": pts, "gf": gf, "ga": ga})

        h = home[["match_date"]].copy(); h["gf"] = home["home_score"]; h["ga"] = home["away_score"]
        a = away[["match_date"]].copy(); a["gf"] = away["away_score"]; a["ga"] = away["home_score"]
        tm = pd.concat([h, a]).sort_values("match_date").tail(5)
        form_pts[t] = int((tm["gf"] > tm["ga"]).sum() * 3 + (tm["gf"] == tm["ga"]).sum())

    table = pd.DataFrame(rows).sort_values(["pts", "gf"], ascending=False).reset_index(drop=True)
    table.index = table.index + 1
    return table, form_pts


def data_storylines(table, form_pts, our_team, max_items=3):
    """Plain-English storylines derived from the live snapshot above.
    Returns (kind, text) pairs, tagged as live since they're computed fresh each time."""
    out = []
    if len(table) == 0:
        return [("live", "The season is just getting under way, too early for a league picture yet.")]
    leader = table.iloc[0]
    out.append(f"{leader['team']} lead the table on {leader['pts']} points from {leader['played']} games.")
    if our_team in table["team"].values:
        rank = int(table.index[table["team"] == our_team][0])
        out.append(f"Manchester City sit {_ordinal(rank)} at this stage of the season.")
    eligible = {t: p for t, p in form_pts.items() if t != our_team}
    if eligible:
        hot_team = max(eligible, key=eligible.get)
        if eligible[hot_team] >= 10:
            out.append(f"{hot_team} are the form side right now, picking up {eligible[hot_team]} points from their last 5.")
    if len(table) >= 18:
        bottom = table.iloc[-1]
        out.append(f"{bottom['team']} prop up the table on {bottom['pts']} points from {bottom['played']} games.")
    return [("live", t) for t in out[:max_items]]
