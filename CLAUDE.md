# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

PitchInstinct is a single-page Streamlit app that acts as a Manchester City manager's decision-support
system, built on StatsBomb's free Premier League 2015/16 event data (competition_id=2, season_id=27).
It mixes descriptive analytics (pressure resistance, xT, passing networks, Elo) with a few genuinely
trained ML models (xG, match outcome, player form) and a synthetic computer-vision pipeline, presented
as FIFA/EA-Sports-styled cards, radars, and pitch maps.

## Commands

```bash
pip install -r requirements.txt
streamlit run app.py
```

There is no test suite, linter, or build step configured. There is no `.git` repo initialized in this
directory (so no git history/branches to inspect).

First run will be slow: `data_loader.py` pulls every match City and (lazily, per-opponent) the
selected opponent played that season via `statsbombpy`, then caches results as parquet files in
`.cache/`. Delete `.cache/` to force a refetch from StatsBomb's open data; otherwise loaders read the
cache straight away.

To populate real player photos (optional): `python fetch_player_photos.py`. Without this, the app
generates initials-on-disc avatars from `player_images.py`.

## Architecture

The app is one big `app.py` (~2000+ lines) with an `if/elif` chain over `st.session_state.page`, where
each branch is a full "page" of the dashboard. The Streamlit sidebar is explicitly hidden via CSS;
navigation lives in a Safari-style horizontal top tab bar driven by the `SECTIONS` dict:

```
🏟️ Match Centre  — Dashboard, Squad, Player Profile, Compare Players, Season Trends, Expected Threat
🎯 Opponent      — Scout Report, Matchup, Who To Stop, Transition, Tactical Degradation
⚙️ Selection     — Build XI, Availability, Pressing Game, Set Pieces
🔮 Predict       — Season Forecast, Live Assistant
🧠 Intelligence  — xG Model, Match Predictor, Form Forecast, Spatial Lab, Vision Lab
❓ Help          — How To Use
```

Two additional pages exist in the `elif` chain but are **not** listed in `SECTIONS` (and are therefore
unreachable via the nav): `"Elo Strength"` and `"Model Validation"`.

Adding a page: add its name to the right `SECTIONS` list, then add a matching `elif page == "...":` block.

Supporting modules, all imported by `app.py` and pure functions over pandas DataFrames (no module-level
side effects beyond caching):

- **data_loader.py** — only module that talks to `statsbombpy`. Defines `OUR_TEAM = "Manchester City"`,
  fetches/caches events, matches, squad stats, and positions to `.cache/*.parquet`. Distinguishes
  City-only events (`get_all_events`, `get_opponent_events`) from full-season per-team data
  (`get_team_full_matches`) needed to scout an opponent's *own* attacking/defensive tendencies.
- **analytics.py** — "Idea"-labelled descriptive metrics: pressure resistance, tactical degradation,
  transition vulnerability, passing-network centrality (networkx betweenness), set-piece analysis, Elo
  ratings, and the best-XI weighted scoring function.
- **features.py** — the largest support module; opponent scouting features (danger men, likely XI,
  pressing triggers, game-state tendencies), FIFA-style attribute/overall derivation
  (`fifa_attributes`, `squad_overalls`, blending a hardcoded FIFA 16 baseline with season performance),
  expected threat (xT) grid construction, Monte Carlo season simulation, live match assistant logic, and
  HTML match-report export.
- **scout.py** — turns an opponent's full-season events into a structured `profile` dict (formation,
  press intensity, attacking flank, etc.) plus `recommend_xi`, which picks City's best XI against that
  specific opponent profile.
- **predictive.py** — the only module with real trained models: gradient-boosted xG (`train_xg_model`),
  multinomial logistic match-outcome model (`train_match_model`), and ridge-regression player-form
  model (`train_form_model`). All are validated out-of-fold/cross-validated and benchmarked against a
  naive baseline before being trusted in the UI.
- **spatial.py** — freeze-frame-based analytics (pitch control via influence functions, defensive shape,
  shot pressure, Voronoi regions) using StatsBomb's per-shot freeze-frame player positions as a stand-in
  for tracking data.
- **spatial_cv.py** — a self-contained OpenCV pipeline (grass-mask segmentation, line detection, player
  blob detection/team assignment via k-means, homography projection) that runs on a *synthetic* generated
  pitch frame (no real broadcast footage) to demonstrate what a CV tracker would feed into `spatial.py`.
- **ui.py** — all HTML/CSS string-building helpers (`inject_theme`, `fut_card`, `stat_card`, `rec_box`,
  radar/pitch chart styling constants). `ui_preglass_backup.py` and `ui_prebroadcast_backup.py` are
  frozen earlier versions kept for reference — don't edit or import them.
- **player_images.py** — resolves a player photo from `assets/players/` by name-slug with alias
  handling, falling back to a generated initials avatar; both paths return a base64 data URI.
- **fetch_player_photos.py** — standalone offline script (not imported by the app) to populate
  `assets/players/` from Wikipedia; run manually, not part of the app's request path.
- **glossary.py** — `GLOSSARY` dict (term → plain-English definition) and `PAGE_TERMS` (page name →
  list of term keys). Rendered by `ui.glossary_box()` as a dropdown on each page.
- **news.py** — curated historical Premier League 2015/16 storylines (`CURATED_STORYLINES`) plus
  `season_snapshot()`, which builds a mini live league table from match scorelines. Always cut off at
  the selected fixture date so the app never reveals results still ahead in that season.

## Key conventions

- All `st.cache_data`/`st.cache_resource` loaders live at the top of `app.py`; expensive joins
  (squad ⨯ positions ⨯ ratings) happen once at module load, not inside page branches. Use
  `st.cache_data` for DataFrames/plain Python values; use `st.cache_resource` for trained sklearn
  objects (the three models in `predictive.py`) — Streamlit serialises the former and shares the
  latter by reference.
- Page branches in `app.py` read three module-level globals set before the `if/elif` chain:
  `opponent` (str), `formation_name` (str key into `FORMATIONS`), and `formation`
  (the corresponding `{"GK":1,"DEF":4,...}` dict). These are populated from the top-bar controls,
  not from the hidden sidebar, and are not passed as arguments into page blocks.
- StatsBomb pitch coordinates: x ∈ [0, 120] (own goal → opponent goal), y ∈ [0, 80] (left → right
  when attacking up the page). City attacks left-to-right in horizontal layouts and bottom-to-top in
  vertical ones. The opponent goal centre is (120, 40).
- `short_name()` in `app.py` maps long official names (e.g. `"Sergio Leonel Agüero del Castillo"`)
  to display names (`"Agüero"`) via `SHORT_NAMES`. Always use it for UI labels; never display raw
  StatsBomb player names directly.
- "Overall" ratings (`squad_overalls` in features.py) are *derived*, not copied from FIFA: a FIFA 16
  baseline rating is adjusted by `0.32 × delta-vs-position-peers × confidence`, capped at ±5. Goalkeepers
  are anchored to baseline since outfield event data can't fairly rate them. Preserve this method if
  touching rating logic — it's explained to end users in-app (Squad page expander).
- `ui.GOLD` is a legacy name; it is currently aliased to mint (`#5FE0CE`), the same as `ui.SKY_HI`.
  The active palette is teal/mint on dark navy. Use `ui.SKY` (teal `#19C4AC`) and `ui.GOLD`/`ui.CYAN`
  (mint `#5FE0CE`/`#34D9C2`) for accent colours; `ui.AMBER` and `ui.RED` for warnings.
- Everything is explicitly "descriptive analytics, not ML" except the three models in `predictive.py`
  and the CV pipeline in `spatial_cv.py` — when adding a feature, keep that distinction clear in any
  UI copy (`ui.legend`/captions reference this).
- Opponent-specific pages depend on the selected `opponent` and trigger `load_opp_full`/
  `scout_opponent`/etc. only when `need_scout` is true for that page (set at the top of `app.py` based
  on `OPP_PAGES` plus an explicit allowlist). New opponent-dependent pages must be added to that gate
  or they'll get stale/missing `profile`/`key_player`.
