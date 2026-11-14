"""
PitchInstinct: Computer Vision Engine  (IMAGE → COORDINATES)

This is the literal computer-vision layer. It takes a single broadcast / tactical
frame (an image) and runs the classical CV pipeline that a tracking provider uses
before any analytics happen:

  1. Pitch segmentation:      HSV grass mask isolates the field of play.
  2. Line detection:          Canny + probabilistic Hough finds pitch lines.
  3. Player detection:        connected-component analysis on the non-grass
                               blobs sitting ON the grass, filtered by size /
                               aspect ratio to player-shaped regions.
  4. Team assignment:         k-means on each detected player's dominant jersey
                               colour → two teams (+ ambiguous / officials).
  5. Homography:              map detected pixel feet-positions onto the
                               120×80 StatsBomb pitch, given four pitch-point
                               correspondences, so CV output speaks the same
                               coordinate language as spatial.py.

It also ships a synthetic tactical-cam generator so the page is fully functional
with no upload: it renders a plausible frame (green pitch, lines, two coloured
teams) and then *detects* on it, demonstrating the full pixels→coordinates loop.

Everything uses OpenCV + NumPy + scikit-learn (KMeans). No deep nets are needed
for this classical pipeline, which keeps it dependency-light and fast.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

PITCH_X, PITCH_Y = 120.0, 80.0


# ═════════════════════════════════════════════════════════════════════════════
# 0.  SYNTHETIC FRAME (so the demo runs with no upload)
# ═════════════════════════════════════════════════════════════════════════════

def synthetic_frame(w=960, h=600, seed=7):
    """Render a simple tactical-cam style frame: green pitch, white lines, two
    teams of coloured circles + a keeper + ball. Returns a BGR uint8 image and
    the ground-truth player list (for optional comparison)."""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    # grass with mild stripe + lighting variation
    for x in range(w):
        base = 90 + 18 * np.sin(x / 55.0)
        img[:, x] = (40, int(120 + base * 0.25), 55)
    img = cv2.GaussianBlur(img, (0, 0), 3)

    white = (235, 235, 235)
    cv2.rectangle(img, (40, 40), (w - 40, h - 40), white, 2)            # touchlines
    cv2.line(img, (w // 2, 40), (w // 2, h - 40), white, 2)             # halfway
    cv2.circle(img, (w // 2, h // 2), 70, white, 2)                     # centre circle
    cv2.rectangle(img, (40, h // 2 - 110), (40 + 130, h // 2 + 110), white, 2)   # left box
    cv2.rectangle(img, (w - 40 - 130, h // 2 - 110), (w - 40, h // 2 + 110), white, 2)  # right box

    teamA = (60, 70, 220)     # red-ish (BGR)
    teamB = (220, 150, 40)    # blue-ish (BGR)
    gt = []

    def place(n, color, xlo, xhi, label):
        for _ in range(n):
            px = int(rng.uniform(xlo, xhi)); py = int(rng.uniform(70, h - 70))
            cv2.circle(img, (px, py), 11, color, -1)
            cv2.circle(img, (px, py), 11, (20, 20, 20), 1)
            gt.append({"px": px, "py": py, "label": label})

    place(10, teamA, 70, w * 0.62, "A")
    place(10, teamB, w * 0.40, w - 70, "B")
    cv2.circle(img, (70, h // 2), 11, (0, 200, 0), -1); gt.append({"px": 70, "py": h // 2, "label": "GK"})
    cv2.circle(img, (w // 2 + 30, h // 2 - 10), 6, (250, 250, 250), -1)  # ball
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, gt


# ═════════════════════════════════════════════════════════════════════════════
# 1.  PITCH SEGMENTATION (grass mask)
# ═════════════════════════════════════════════════════════════════════════════

def grass_mask(img_bgr):
    """HSV threshold for green grass → binary mask, morphologically cleaned."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # broad green band; tolerant to lighting/stripes
    lo = np.array([30, 35, 35]); hi = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask


def pitch_coverage(mask):
    return round(float((mask > 0).mean()) * 100, 1)


# ═════════════════════════════════════════════════════════════════════════════
# 2.  LINE DETECTION (Canny + Hough)
# ═════════════════════════════════════════════════════════════════════════════

def detect_lines(img_bgr, mask=None):
    """Probabilistic Hough line segments restricted to the pitch region."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(gray, 60, 160, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=70, maxLineGap=25)
    segs = []
    if lines is not None:
        for l in lines[:, 0, :]:
            segs.append(tuple(int(v) for v in l))
    return segs, edges


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PLAYER DETECTION (blobs on grass)
# ═════════════════════════════════════════════════════════════════════════════

def detect_players(img_bgr, mask):
    """Find player-shaped non-grass blobs that sit ON the field of play.

    Improvements over a naive blob search:
      • restrict to the largest grass region (the pitch) so stands / crowd /
        advertising boards above the pitch are excluded;
      • require a minimum colourfulness so dull crowd speckle and white lines
        are ignored (players wear saturated kit colours);
      • rank by a 'player-likeness' score and cap the count, so a noisy frame
        can't report 40 players.
    """
    h, w = mask.shape
    grass = (mask > 0).astype(np.uint8)

    # field of play = largest filled grass contour (drops stands/ads above pitch)
    cnts, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    field = np.zeros_like(grass)
    if cnts:
        big = max(cnts, key=cv2.contourArea)
        cv2.drawContours(field, [big], -1, 1, -1)
    else:
        field = grass.copy()
    field = cv2.erode(field, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    not_grass = ((grass == 0).astype(np.uint8)) * 255
    candidates = cv2.bitwise_and(not_grass, not_grass, mask=field)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    n, labels, stats, cent = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    img_area = h * w
    gmask = (mask > 0)
    scored = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < img_area * 0.00020 or area > img_area * 0.013:
            continue
        aspect = bh / (bw + 1e-6)
        if aspect < 0.8 or aspect > 3.8:         # players are tallish blobs
            continue
        fill = area / float(bw * bh + 1e-6)
        if fill < 0.35:                          # sparse blob = line fragment / noise
            continue
        cx = x + bw / 2.0
        foot_y = y + bh
        # a real player STANDS on the grass: require grass just beneath the blob
        fy = min(int(foot_y) + 4, h - 1); fx = int(np.clip(cx, 0, w - 1))
        below = gmask[max(fy-3,0):min(fy+4,h), max(fx-4,0):min(fx+5,w)]
        if below.size == 0 or below.mean() < 0.40:
            continue
        # and the blob must be an ISLAND IN the grass: its immediate surround is
        # mostly green. Crowd / dugout / ad-board blobs are surrounded by non-grass.
        pad = max(3, int(0.45 * max(bw, bh)))
        rx0, ry0 = max(x-pad,0), max(y-pad,0); rx1, ry1 = min(x+bw+pad,w), min(y+bh+pad,h)
        outer = gmask[ry0:ry1, rx0:rx1]
        inner = gmask[y:min(y+bh,h), x:min(x+bw,w)]
        ring_area = outer.size - inner.size
        if ring_area <= 0 or (outer.sum() - inner.sum()) / ring_area < 0.55:
            continue
        patch = img_bgr[y:y + max(bh // 2, 1), x:x + bw]   # jersey = upper half
        if patch.size == 0:
            continue
        dom = patch.reshape(-1, 3).mean(axis=0)
        chroma = float(dom.max() - dom.min())
        if chroma < 26:                          # skip dull/grey crowd & lines
            continue
        score = chroma + 30 * max(0.0, 1 - abs(aspect - 1.7) / 2.5)
        scored.append((score, {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
                               "cx": float(cx), "foot_x": float(cx), "foot_y": float(foot_y),
                               "b": float(dom[0]), "g": float(dom[1]), "r": float(dom[2])}))

    scored.sort(key=lambda t: -t[0])
    # non-maximum suppression: drop near-duplicate detections of the same player
    min_dist = max(12.0, 0.9 * np.median([d["h"] for _, d in scored])) if scored else 12.0
    kept = []
    for _, d in scored:
        if all((d["cx"] - k["cx"])**2 + (d["foot_y"] - k["foot_y"])**2 > min_dist**2 for k in kept):
            kept.append(d)
        if len(kept) >= 22:                      # a match has at most 22 players on the pitch
            break
    return kept


# ═════════════════════════════════════════════════════════════════════════════
# 4.  TEAM ASSIGNMENT (k-means on jersey colour)
# ═════════════════════════════════════════════════════════════════════════════

def assign_teams(players, k=2):
    """Cluster detected players by dominant jersey colour into 2 teams."""
    if len(players) < 2:
        for p in players:
            p["team"] = 0
        return players, None
    feats = np.array([[p["r"], p["g"], p["b"]] for p in players], dtype=float)
    km = KMeans(n_clusters=min(k, len(players)), n_init=10, random_state=42).fit(feats)
    for p, lab in zip(players, km.labels_):
        p["team"] = int(lab)
    return players, km.cluster_centers_


# ═════════════════════════════════════════════════════════════════════════════
# 5.  HOMOGRAPHY (pixels → pitch coordinates)
# ═════════════════════════════════════════════════════════════════════════════

def default_correspondences(img_shape):
    """Four sensible image↔pitch point pairs for a tactical-cam style frame.
    For the synthetic frame these line up with the drawn touchlines/boxes.
    src = image pixels (TL, TR, BR, BL of the visible pitch);
    dst = pitch coords (0,0)-(120,80)."""
    h, w = img_shape[:2]
    margin_x, margin_y = 40, 40
    src = np.float32([[margin_x, margin_y], [w - margin_x, margin_y],
                      [w - margin_x, h - margin_y], [margin_x, h - margin_y]])
    dst = np.float32([[0, 0], [PITCH_X, 0], [PITCH_X, PITCH_Y], [0, PITCH_Y]])
    return src, dst


def compute_homography(src, dst):
    return cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))


def project_players(players, H):
    """Map each player's foot pixel to pitch coordinates via homography H."""
    if not players:
        return players
    pts = np.float32([[[p["foot_x"], p["foot_y"]]] for p in players])
    proj = cv2.perspectiveTransform(pts, H)[:, 0, :]
    for p, (px, py) in zip(players, proj):
        p["pitch_x"] = float(np.clip(px, 0, PITCH_X))
        p["pitch_y"] = float(np.clip(py, 0, PITCH_Y))
    return players


# ═════════════════════════════════════════════════════════════════════════════
# 6.  ONE-CALL PIPELINE + ANNOTATED OVERLAY
# ═════════════════════════════════════════════════════════════════════════════

def analyze_frame(img_bgr, src=None, dst=None):
    """Run the full pipeline on one BGR image. Returns a dict with the mask,
    line segments, detected+team-assigned+projected players, and metrics."""
    mask = grass_mask(img_bgr)
    segs, edges = detect_lines(img_bgr, mask)
    players = detect_players(img_bgr, mask)
    players, centers = assign_teams(players)
    if src is None or dst is None:
        src, dst = default_correspondences(img_bgr.shape)
    H = compute_homography(src, dst)
    players = project_players(players, H)
    return {
        "mask": mask, "edges": edges, "lines": segs, "players": players,
        "team_centers": centers, "H": H, "src": src, "dst": dst,
        "metrics": {
            "pitch_coverage_pct": pitch_coverage(mask),
            "n_lines": len(segs),
            "n_players": len(players),
            "team0": int(sum(1 for p in players if p.get("team") == 0)),
            "team1": int(sum(1 for p in players if p.get("team") == 1)),
        },
    }


def annotate(img_bgr, result):
    """Draw detections back onto the frame: lines (cyan), player boxes coloured
    by team, foot points. Returns an RGB image ready for st.image / matplotlib."""
    out = img_bgr.copy()
    for (x1, y1, x2, y2) in result["lines"]:
        cv2.line(out, (x1, y1), (x2, y2), (245, 245, 60), 2)
    team_colors = {0: (60, 70, 235), 1: (235, 170, 50)}
    for p in result["players"]:
        c = team_colors.get(p.get("team", 0), (200, 200, 200))
        cv2.rectangle(out, (p["x"], p["y"]), (p["x"] + p["w"], p["y"] + p["h"]), c, 2)
        cv2.circle(out, (int(p["foot_x"]), int(p["foot_y"])), 3, (255, 255, 255), -1)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def players_to_frame_df(result):
    """Convert CV-detected, pitch-projected players into the same schema
    spatial.py's freeze-frame functions expect (x, y, teammate, is_gk)."""
    import pandas as pd
    rows = []
    for p in result["players"]:
        if "pitch_x" not in p:
            continue
        rows.append({"x": p["pitch_x"], "y": p["pitch_y"],
                     "teammate": p.get("team", 0) == 0,
                     "is_gk": False, "position": "", "player": ""})
    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
# MANAGER READS: turn a detected freeze-frame into actionable tactical calls
# Works in normalised field space (robust to camera angle / homography error).
# ═════════════════════════════════════════════════════════════════════════════

def field_bounds(mask):
    """Bounding box of the largest grass region (the field of play)."""
    grass = (mask > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(grass, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        h, w = mask.shape
        return 0, 0, w, h
    x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
    return x, y, w, h


COL_NAMES = ["left", "central", "right"]
ROW_NAMES = ["near", "middle", "far"]

def tactical_reads(players, mask, our_team=0):
    """From detected players (with team + foot point), produce manager reads:
    a 3x3 overload grid, compactness/width per team, the biggest open space,
    and plain-English takeaways. our_team = 0 or 1 (which colour cluster is us)."""
    import numpy as np
    fx, fy, fw, fh = field_bounds(mask)
    pts = [p for p in players if "team" in p]
    if fw <= 0 or fh <= 0 or len(pts) < 4:
        return None

    def norm(p):
        nx = np.clip((p["foot_x"] - fx) / fw, 0, .999)   # 0=left .. 1=right
        ny = np.clip((p["foot_y"] - fy) / fh, 0, .999)   # 0=near(top).. 1=far
        return nx, ny

    grid_us = np.zeros((3, 3), int); grid_them = np.zeros((3, 3), int)
    us_xy, them_xy = [], []
    for p in pts:
        nx, ny = norm(p); col = int(nx * 3); row = int(ny * 3)
        if p["team"] == our_team:
            grid_us[row, col] += 1; us_xy.append((nx, ny))
        else:
            grid_them[row, col] += 1; them_xy.append((nx, ny))
    us_xy = np.array(us_xy); them_xy = np.array(them_xy)

    def compactness(a):
        if len(a) < 2: return 0.0
        d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
        return float(d[np.triu_indices(len(a), 1)].mean())
    def width(a):
        return float(a[:, 0].max() - a[:, 0].min()) if len(a) else 0.0

    adv = grid_us - grid_them
    # biggest space = grid cell with the fewest total players
    total = grid_us + grid_them
    sr, sc = np.unravel_index(np.argmin(total), total.shape)

    takeaways = []
    # our biggest overload
    mr, mc = np.unravel_index(np.argmax(adv), adv.shape)
    if adv[mr, mc] >= 2:
        takeaways.append(("good", f"You have a <b>+{adv[mr,mc]} overload</b> in the "
            f"{ROW_NAMES[mr]}-{COL_NAMES[mc]} zone, play through it."))
    # their biggest overload
    tr, tc = np.unravel_index(np.argmin(adv), adv.shape)
    if adv[tr, tc] <= -2:
        takeaways.append(("warn", f"They outnumber you by <b>{-adv[tr,tc]}</b> in the "
            f"{ROW_NAMES[tr]}-{COL_NAMES[tc]} zone, shift across or drop a man in."))
    # shape
    cu, ct = compactness(us_xy), compactness(them_xy)
    if cu > ct * 1.15:
        takeaways.append(("warn", "Your shape is more <b>stretched</b> than theirs, tighten the gaps between lines."))
    elif ct > cu * 1.15:
        takeaways.append(("good", "You're more <b>compact</b> than them, keep the block tight and force them wide."))
    # biggest space
    takeaways.append(("info", f"The largest <b>open space</b> is the {ROW_NAMES[sr]}-{COL_NAMES[sc]} zone, "
                              f"attack it on the ball, cover it off it."))

    return {
        "grid_us": grid_us, "grid_them": grid_them, "adv": adv,
        "compact_us": round(cu, 3), "compact_them": round(ct, 3),
        "width_us": round(width(us_xy), 2), "width_them": round(width(them_xy), 2),
        "space_zone": f"{ROW_NAMES[sr]}-{COL_NAMES[sc]}",
        "n_us": len(us_xy), "n_them": len(them_xy),
        "takeaways": takeaways,
    }


# ═════════════════════════════════════════════════════════════════════════════
# 8.  MANAGER READ: turn a mapped frame into tactical decisions
# ═════════════════════════════════════════════════════════════════════════════

def tactical_reads(frame_df, our_team_is_A=True):
    """Manager-facing read of one mapped frame: zonal overloads, block shape,
    and the biggest exploitable space. Coordinates are mirrored so the read is
    always from OUR attacking perspective (we attack toward high x = right)."""
    import pandas as pd
    if frame_df is None or len(frame_df) == 0:
        return None
    df = frame_df.copy()
    df["ours"] = (df["teammate"] == bool(our_team_is_A))
    # mirror so we always attack to the right and "our left" is consistent
    if not our_team_is_A:
        df["x"] = 120 - df["x"]; df["y"] = 80 - df["y"]
    ours = df[df["ours"]]; theirs = df[~df["ours"]]
    if len(ours) == 0 or len(theirs) == 0:
        return None

    thirds = ["Defensive", "Middle", "Attacking"]
    channels = ["left", "centre", "right"]
    def zi(x, y):
        c = 0 if x < 40 else (1 if x < 80 else 2)
        r = 0 if y < 26.7 else (1 if y < 53.3 else 2)
        return r, c
    grid = np.zeros((3, 3))
    for _, p in ours.iterrows():
        r, c = zi(p["x"], p["y"]); grid[r, c] += 1
    for _, p in theirs.iterrows():
        r, c = zi(p["x"], p["y"]); grid[r, c] -= 1

    # biggest overload (prefer middle/attacking thirds) and biggest deficit (our half)
    best = None; worst = None
    for c in range(3):
        for r in range(3):
            v = grid[r, c]
            if v > 0 and c >= 1 and (best is None or v > best[0]):
                best = (v, channels[r], thirds[c])
            if v < 0 and c <= 1 and (worst is None or v < worst[0]):
                worst = (v, channels[r], thirds[c])

    width = float(ours["y"].max() - ours["y"].min())
    depth = float(ours["x"].max() - ours["x"].min())
    line_height = float(ours["x"].min())          # our deepest man = back line
    compact = "compact" if width < 34 else ("balanced" if width < 50 else "stretched")

    # biggest open space we control in their half: grid of cells, nearest-player
    cells = []
    for gx in np.linspace(66, 114, 7):
        for gy in np.linspace(8, 72, 7):
            do = min(((ours["x"]-gx)**2 + (ours["y"]-gy)**2) ** 0.5) if len(ours) else 99
            dt = min(((theirs["x"]-gx)**2 + (theirs["y"]-gy)**2) ** 0.5) if len(theirs) else 99
            cells.append((dt - do, gx, gy))        # we're closer than them -> our space
        # (positive = space we can attack into)
    space = max(cells, key=lambda t: t[0]) if cells else None

    recs = []
    if best:
        recs.append(("good", f"You have a <b>+{int(best[0])} overload {best[1]} in the {best[2]} third</b>, "
                             f"play through it / switch the ball there."))
    if worst:
        recs.append(("warn", f"They outnumber you <b>{int(abs(worst[0]))} {worst[1]} in the {worst[2]} third</b>, "
                             f"shift across or drop a man in to cover."))
    recs.append(("info", f"Your block is <b>{compact}</b> ({width:.0f}m wide, {depth:.0f}m deep). "
                         f"Back line sitting around x={line_height:.0f}/120."))
    if space and space[0] > 6:
        side = "right" if space[2] > 53 else ("left" if space[2] < 27 else "centre")
        recs.append(("good", f"Biggest free space is <b>high on the {side}</b>, a runner in there is unmarked."))

    return {"grid": grid, "thirds": thirds, "channels": channels,
            "ours": len(ours), "theirs": len(theirs), "width": width, "depth": depth,
            "compact": compact, "line_height": line_height, "recs": recs,
            "space_xy": (space[1], space[2]) if space else None}
