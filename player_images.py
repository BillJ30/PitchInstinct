"""
PitchInstinct: Player image system

Gives every player a face on their card, with two layers:

  1. REAL PHOTOS (optional). Drop an image at  assets/players/<slug>.<ext>
     (png / jpg / jpeg / webp) and the card uses it automatically. Slugs are
     forgiving: "aguero.png", "sergio-aguero.png", and the full StatsBomb name
     all resolve to the same player via an alias table. Run
     `python fetch_player_photos.py` to populate the folder from a public source.

  2. GENERATED AVATARS (always work, fully offline, no external assets). If no
     photo file exists, we render a clean monogram avatar: player initials on a
     position-tinted gradient with a subtle ring, cached to .cache/avatars so
     it's only drawn once. This is the default so the app looks complete out of
     the box.

Both layers return a value usable directly in an <img src="..."> inside Streamlit
markdown: a data URI for generated avatars, or a relative file path for photos.
"""

import os, re, base64, hashlib, unicodedata
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(HERE, "assets", "players")
CACHE_DIR = os.path.join(HERE, ".cache", "avatars")
os.makedirs(PHOTO_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

PHOTO_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Position-based avatar tints (matches the app's floodlit palette)
_POS_TINT = {
    "GK":  ((28, 64, 48),  (63, 215, 126)),    # green keeper
    "DEF": ((14, 30, 56),  (108, 171, 221)),   # sky
    "MID": ((20, 24, 60),  (143, 196, 237)),   # light blue
    "FWD": ((16, 34, 58),  (110, 190, 240)),   # sky attacker (no gold)
    "SUB": ((20, 30, 50),  (126, 147, 181)),
}

# ── name handling ─────────────────────────────────────────────────────────────

def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))

def slug(name: str) -> str:
    """A filename-safe slug from a player name (accents removed, lowercased)."""
    s = _strip_accents(name).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s

# Short, well-known names → used both for nice initials and as photo-file aliases.
# Keyed by a distinctive token from the full StatsBomb name.
_SHORT = {
    "Sergio Leonel Agüero del Castillo": "Sergio Agüero",
    "David Josué Jiménez Silva":         "David Silva",
    "Gnégnéri Yaya Touré":               "Yaya Touré",
    "Nicolás Hernán Otamendi":           "Nicolás Otamendi",
    "Pablo Javier Zabaleta Girod":       "Pablo Zabaleta",
    "Jesús Navas González":              "Jesús Navas",
    "Martín Gastón Demichelis":          "Martín Demichelis",
    "Kelechi Promise Iheanacho":         "Kelechi Iheanacho",
    "Wilfried Guemiand Bony":            "Wilfried Bony",
    "Wilfredo Daniel Caballero":         "Willy Caballero",
    "Fernando Francisco Reges":          "Fernandinho",
    "Fernando Luiz Rosa":                "Fernando",
    "Fernando Luiz Roza":                "Fernando",
    "Manuel García Alonso":              "Manu García",
    "Bersant Celina":                    "Bersant Celina",
    "Patrick Roberts":                   "Patrick Roberts",
}

def display_name(name: str) -> str:
    return _SHORT.get(name, name)

def _initials(name: str) -> str:
    """Sensible 2-letter initials from the SHORT name (e.g. Agüero -> SA)."""
    short = display_name(name)
    parts = [p for p in _strip_accents(short).split() if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def _aliases(name: str):
    """All slugs a photo file might reasonably be named after this player."""
    al = {slug(name), slug(display_name(name))}
    short = display_name(name)
    parts = _strip_accents(short).split()
    if parts:
        al.add(slug(parts[-1]))               # surname only, e.g. "aguero"
    return {a for a in al if a}

# ── photo lookup ──────────────────────────────────────────────────────────────

def find_photo(name: str):
    """Return a real photo as a base64 data URI if one exists, else None.

    We base64-encode rather than returning a path because Streamlit does not
    serve arbitrary local files referenced by a raw <img src="..."> tag; a data
    URI always renders. Images are resized to a sensible max so cards stay light.
    """
    for a in _aliases(name):
        for ext in PHOTO_EXTS:
            p = os.path.join(PHOTO_DIR, a + ext)
            if os.path.exists(p):
                return _photo_data_uri(p)
    return None


def _photo_data_uri(path: str, box: int = 256) -> str:
    """Load a real photo, center-crop to square, downscale, return a data URI.
    Falls back to raw bytes if PIL can't process it."""
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
        im = im.resize((box, box), Image.LANCZOS)
        buf = BytesIO()
        im.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        with open(path, "rb") as f:
            raw = f.read()
        mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
        return f"data:{mime};base64," + base64.b64encode(raw).decode()

# ── avatar generation ─────────────────────────────────────────────────────────

def _font(size: int):
    for cand in [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]:
        if os.path.exists(cand):
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _gradient(size, c0, c1):
    """Vertical gradient image."""
    base = Image.new("RGB", (1, size), 0)
    for y in range(size):
        t = y / max(size - 1, 1)
        base.putpixel((0, y), tuple(int(c0[i] + (c1[i] - c0[i]) * t) for i in range(3)))
    return base.resize((size, size))

def generate_avatar(name: str, pos_group: str = "MID", size: int = 256) -> str:
    """Render (and cache) a monogram avatar; return a data URI."""
    key = hashlib.md5(f"{slug(name)}|{pos_group}|{size}|v2".encode()).hexdigest()[:16]
    cache_path = os.path.join(CACHE_DIR, key + ".png")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()

    c0, c1 = _POS_TINT.get(pos_group, _POS_TINT["MID"])
    S = size * 2  # supersample for crisp edges
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # circular gradient background
    grad = _gradient(S, c0, c1).convert("RGBA")
    mask = Image.new("L", (S, S), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, S, S], fill=255)
    img.paste(grad, (0, 0), mask)

    d = ImageDraw.Draw(img)
    # subtle inner highlight (top-left) for depth
    hi = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(hi).ellipse([S*0.08, S*0.05, S*0.95, S*0.92], fill=(255, 255, 255, 26))
    hi = hi.filter(ImageFilter.GaussianBlur(S * 0.06))
    img = Image.alpha_composite(img, hi)
    d = ImageDraw.Draw(img)

    # ring
    ring = c1
    d.ellipse([S*0.03, S*0.03, S*0.97, S*0.97], outline=(ring[0], ring[1], ring[2], 230),
              width=max(2, int(S*0.018)))

    # initials
    txt = _initials(name)
    f = _font(int(S * 0.40))
    bb = d.textbbox((0, 0), txt, font=f)
    tw, th = bb[2] - bb[0], bb[3] - bb[1]
    d.text(((S - tw) / 2 - bb[0], (S - th) / 2 - bb[1]), txt, font=f, fill=(255, 255, 255, 235))

    img = img.resize((size, size), Image.LANCZOS)
    img.save(cache_path)
    with open(cache_path, "rb") as fp:
        return "data:image/png;base64," + base64.b64encode(fp.read()).decode()

# ── public API ────────────────────────────────────────────────────────────────

def player_image(name: str, pos_group: str = "MID", size: int = 256):
    """Return an <img src>-ready value: a real photo path if present, else an
    auto-generated avatar data URI. Always returns something usable."""
    photo = find_photo(name)
    if photo:
        return photo
    return generate_avatar(name, pos_group or "MID", size)

def has_real_photo(name: str) -> bool:
    for a in _aliases(name):
        for ext in PHOTO_EXTS:
            if os.path.exists(os.path.join(PHOTO_DIR, a + ext)):
                return True
    return False
