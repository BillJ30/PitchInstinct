# PitchInstinct

**A match-preparation decision support app for a football manager, built with Python and Streamlit.**

PitchInstinct puts you in the Manchester City manager's seat for the 2015/16 Premier League season. Pick the next opponent and a formation, and the app scouts that opponent, helps you choose a starting XI, and forecasts how the match and the season could go. Everything runs on StatsBomb's free open event data.

---

## What the app does

The app is organised into six sections, reached from the tab bar at the top. The **Opponent** and **Formation** controls drive every page.

| Section | Pages | What you get |
|---|---|---|
| 🏟️ **Match Centre** | Dashboard, Squad, Player Profile, Compare Players, Season Trends, Expected Threat | City's season so far, player cards and ratings, side-by-side comparisons, form over time and an expected threat (xT) map of where possession creates danger |
| 🎯 **Opponent** | Scout Report, Matchup, Who To Stop, Transition, Tactical Degradation | The opponent's likely XI, danger men, pressing triggers and set-piece habits, their key passer from a passing network, how exposed they are in transition, and how their shape fades late in games |
| ⚙️ **Selection** | Build XI, Availability, Pressing Game, Set Pieces | A recommended XI against this specific opponent and formation, squad availability, and pressing and set-piece plans |
| 🔮 **Predict** | Season Forecast, Live Assistant | A Monte Carlo simulation of the rest of the season, plus an in-match assistant that suggests changes as the game state shifts |
| 🧠 **Intelligence** | xG Model, Match Predictor, Form Forecast, Spatial Lab, Vision Lab | The machine learning models, pitch control analysis from shot freeze frames, and a computer vision demo |
| ❓ **Help** | How To Use | A guide to the app, with a plain-English glossary on every page |

The app only uses information available **before** the selected fixture date, so it never gives away results that were still to come in that season.

## How it works

**Data.** Event data for every City match and every opponent's season comes from [StatsBomb Open Data](https://github.com/statsbomb/open-data) through `statsbombpy`. It is downloaded on first use and cached locally as parquet files.

**Descriptive analytics.** Pressure resistance, expected threat (xT), passing-network centrality (`networkx`), transition vulnerability, tactical degradation, Elo team strength and opponent scouting profiles.

**Machine learning** (in `predictive.py`). Each model is cross-validated and checked against a simple baseline before the app relies on it:
- **Expected goals (xG):** gradient boosting on shot features
- **Match outcome:** multinomial logistic regression (win / draw / loss)
- **Player form:** ridge regression forecasting upcoming performance

**Player ratings.** Each player's overall rating starts from a FIFA 16 baseline and is adjusted by how they performed against positional peers this season, capped at ±5. Goalkeepers stay at their baseline, since event data can't rate them fairly.

**Spatial and vision.** Pitch control, defensive shape and Voronoi regions are built from StatsBomb's shot freeze frames, standing in for tracking data. The Vision Lab runs an OpenCV pipeline (pitch segmentation, line detection, player detection, team assignment with k-means, homography) on a generated pitch image. It shows what a real tracking system would feed into the spatial analysis.

## Run it locally

You need **Python 3.10 or newer**.

```bash
git clone https://github.com/BillJ30/PitchInstinct.git
cd PitchInstinct
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

**First run is slow.** The app downloads City's full season from StatsBomb, which takes a minute or two. Each new opponent you select downloads on first use. Everything is cached in a `.cache/` folder, so later runs start quickly. Delete that folder to force a fresh download.

**Optional player photos.** Run `python fetch_player_photos.py` to download player photos into `assets/players/`. Without them, the app shows initials avatars.

## Project structure

| File | Purpose |
|---|---|
| `app.py` | The Streamlit app: layout, navigation and every page |
| `data_loader.py` | Downloads and caches StatsBomb data (the only module that talks to StatsBomb) |
| `analytics.py` | Pressure resistance, passing networks, transitions, Elo, best-XI scoring |
| `features.py` | Opponent scouting, player ratings, xT, season simulation, live assistant |
| `scout.py` | Opponent profiles and XI recommendation against a given opponent |
| `predictive.py` | xG, match outcome and player form models |
| `spatial.py` | Pitch control, defensive shape and Voronoi analysis from freeze frames |
| `spatial_cv.py` | OpenCV computer vision demo |
| `ui.py` | Styling and card, radar and pitch helpers |
| `player_images.py`, `fetch_player_photos.py` | Player photos and initials avatars |
| `glossary.py`, `news.py` | In-app glossary and season storylines |

## Built with

Python · Streamlit · pandas · NumPy · scikit-learn · SciPy · Plotly · Matplotlib · mplsoccer · networkx · OpenCV · statsbombpy

## Data credit

Event data provided by [StatsBomb](https://statsbomb.com/) through their free Open Data release.

---

Built by **Mohammad Bilal**, MSc Business Analytics, UCD Michael Smurfit Graduate Business School.
