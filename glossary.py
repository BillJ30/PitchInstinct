"""
PitchInstinct: Glossary
Plain-English definitions for the football/analytics jargon used across the app.
Looked up by ui.glossary_box() and shown as a dropdown on each page.
"""

GLOSSARY = {
    "xg": "Expected Goals (xG): the probability a shot should have resulted in a goal, "
          "based on thousands of historical shots from similar positions and situations.",
    "xt": "Expected Threat (xT): the increase in goal-scoring danger created by moving the "
          "ball from one pitch zone to another, even without a shot.",
    "elo": "Elo Rating: a self-adjusting team-strength score (borrowed from chess) that rises "
           "after wins and falls after losses, weighted by how surprising the result was.",
    "ppda": "PPDA (Passes Per Defensive Action): opponent passes allowed before a tackle, "
            "interception or pressure; lower PPDA = a more aggressive press.",
    "press_resistance": "Press Resistance: how well a player keeps possession when challenged "
                         "close by an opponent, measured as pass-completion % under pressure.",
    "pr_score": "PR Score: a single 0-100 press-resistance rating blending a player's pass "
                "accuracy under pressure with how often they're pressured at all.",
    "betweenness_centrality": "Betweenness Centrality: how often a player sits on the shortest "
                               "passing path between two teammates; a high score means play "
                               "regularly flows through him.",
    "passing_network": "Passing Network: a map of who passes to whom, built from real match "
                        "events, used to find a team's creative hub.",
    "pitch_control": "Pitch Control: which team would win the ball first in any patch of grass "
                      "right now, estimated from every player's position and proximity.",
    "compactness": "Defensive Compactness: how tightly packed a team's defenders are; a lower "
                   "number means a tighter, harder-to-play-through block.",
    "line_height": "Line Height: how far up the pitch a team's defensive line sits; a high "
                    "line risks space in behind but compresses the pitch for pressing.",
    "freeze_frame": "Freeze-Frame: a snapshot of every visible player's exact position at the "
                     "instant of a shot, the same signal a video tracking system would capture.",
    "game_state": "Game State: whether a team is Leading, Level or Trailing; behaviour (press, "
                  "risk-taking, tempo) often changes sharply with the scoreline.",
    "pressing_trigger": "Pressing Trigger: the situations (e.g. a poor first touch, a sideways "
                         "pass) that cue a team to press immediately.",
    "tactical_degradation": "Tactical Degradation: whether a team's shape (depth, width) "
                             "collapses or drops in the closing stages, often from fatigue.",
    "transition": "Transition: the few seconds right after winning or losing the ball, when "
                   "defensive shape is most disorganised and space is easiest to exploit.",
    "overall_rating": "Overall (Derived): a 0-100 rating built from a preseason baseline grade, "
                     "adjusted by real season performance (see Squad page for the formula).",
    "confidence": "Confidence: how many appearances a player has this season; low-minute "
                  "players get a smaller rating adjustment so one good game doesn't skew it.",
    "danger_man": "Danger Man: an opponent ranked by combined goals, assists and chances "
                   "created; the players most likely to hurt you if left unmarked.",
    "likely_xi": "Likely XI: the eleven players who've started most often this season for that "
                  "opponent, used to anticipate their team selection.",
    "tactical_preset": "Tactical Preset: a one-click manager style (e.g. Possession, "
                        "Counter-Attack) that sets the selection weights for Build XI automatically.",
    "load": "Load (Workload): a fatigue proxy combining a player's recent and season minutes; "
            "not medical data, just an indicator for rotation risk.",
    "booking_risk": "Booking Risk: a player on 4+ yellow cards this season, one booking away "
                     "from a suspension under the league's accumulation rule.",
    "risk_appetite": "Risk Appetite: how attacking vs. defensive an approach the manager wants "
                      "to take into a fixture; more attacking trades defensive solidity for goal threat.",
    "set_piece": "Set Piece: a dead-ball situation (corner, free kick, throw-in) where play "
                 "restarts from a fixed position rather than open play.",
    "key_pass": "Key Pass: a pass that directly creates a shot for a teammate, regardless of "
                "whether the shot is scored.",
    "possession": "Possession %: the share of all passes in a match completed by one team, "
                   "used as a rough proxy for who controlled the ball.",
    "monte_carlo": "Monte Carlo Simulation: replaying an entire season hundreds of times with "
                    "randomised match outcomes (weighted by Elo) to estimate finishing odds.",
    "calibration": "Calibration: whether a model's predicted probabilities match real-world "
                    "frequency (e.g. among shots given a 30% chance, about 30% should be goals).",
    "form": "Form: a player's or team's rolling recent performance (typically a 3-5 game "
            "window), used to read momentum rather than season-long averages.",
}

# Which glossary terms are relevant on each page, keyed by app.py's `page` string.
PAGE_TERMS = {
    "Scout Report": ["ppda", "set_piece", "game_state", "danger_man", "likely_xi", "pressing_trigger"],
    "Tactical Degradation": ["tactical_degradation", "compactness", "line_height"],
    "Pressure Resistance": ["press_resistance", "pr_score"],
    "Pressing": ["ppda", "pressing_trigger"],
    "Set Pieces": ["set_piece", "key_pass"],
    "Elo Strength": ["elo"],
    "Expected Threat": ["xt", "possession"],
    "Availability": ["load", "booking_risk"],
    "xG Model": ["xg", "freeze_frame", "calibration"],
    "Match Predictor": ["form", "calibration"],
    "Form Forecast": ["form"],
    "Spatial Lab": ["pitch_control", "compactness", "line_height", "freeze_frame"],
    "Season Forecast": ["monte_carlo", "elo", "form"],
    "Live Assistant": ["game_state", "risk_appetite"],
    "Player Profile": ["overall_rating", "confidence", "form"],
}
