"""
fetch_player_photos.py: OPTIONAL, download real player photos into
assets/players/ so the cards show faces instead of monogram avatars.

This is opt-in and runs on YOUR machine (it needs internet access, which the
app itself does not). It queries the public Wikipedia/Wikimedia REST API for
each Manchester City player's lead image and saves it as a slugged PNG/JPG.

    python fetch_player_photos.py

Notes
-----
• The app works fully without this, generated avatars are the default.
• Image licensing varies per file on Wikimedia (many are CC-BY / CC-BY-SA, some
  are non-free). You are responsible for confirming a given image is cleared for
  your use. This script is a convenience, not a rights grant.
• Anything it can't find is simply skipped; those players keep their avatar.
• You can always drop your own image manually at
  assets/players/<surname>.png  (e.g. assets/players/aguero.png).
"""

import os, sys, time, json, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "assets", "players")
os.makedirs(OUT, exist_ok=True)

# Map each player to (slug_filename, wikipedia_page_title)
PLAYERS = {
    "aguero":      "Sergio Agüero",
    "silva":       "David Silva",
    "de-bruyne":   "Kevin De Bruyne",
    "kompany":     "Vincent Kompany",
    "sterling":    "Raheem Sterling",
    "toure":       "Yaya Touré",
    "fernandinho": "Fernandinho",
    "iheanacho":   "Kelechi Iheanacho",
    "bony":        "Wilfried Bony",
    "kolarov":     "Aleksandar Kolarov",
    "nasri":       "Samir Nasri",
    "sagna":       "Bacary Sagna",
    "zabaleta":    "Pablo Zabaleta",
    "hart":        "Joe Hart",
    "otamendi":    "Nicolás Otamendi",
    "delph":       "Fabian Delph",
    "clichy":      "Gaël Clichy",
    "demichelis":  "Martín Demichelis",
    "navas":       "Jesús Navas",
    "caballero":   "Willy Caballero",
    "mangala":     "Eliaquim Mangala",
    "fernando":    "Fernando Reges",
    "sane":        "Leroy Sané",
    "roberts":     "Patrick Roberts",
}

UA = {"User-Agent": "PitchInstinct/1.0 (personal football analytics app)"}
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/"


def fetch_one(slug_name, title):
    url = SUMMARY + urllib.parse.quote(title.replace(" ", "_"))
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.load(r)
    except Exception as e:
        return f"  · {title:22s}: lookup failed ({e})"

    img = (data.get("thumbnail") or {}).get("source") or \
          (data.get("originalimage") or {}).get("source")
    if not img:
        return f"  · {title:22s}: no image on page, skipped"

    ext = ".png" if img.lower().endswith(".png") else ".jpg"
    dest = os.path.join(OUT, slug_name + ext)
    try:
        req = urllib.request.Request(img, headers=UA)
        with urllib.request.urlopen(req, timeout=20) as r:
            blob = r.read()
        with open(dest, "wb") as f:
            f.write(blob)
        return f"  ✓ {title:22s} -> assets/players/{slug_name}{ext}  ({len(blob)//1024} KB)"
    except Exception as e:
        return f"  · {title:22s}: download failed ({e})"


def main():
    print(f"Fetching player photos into {OUT}\n(licensing varies per image, verify before any public use)\n")
    ok = 0
    for slug_name, title in PLAYERS.items():
        line = fetch_one(slug_name, title)
        print(line)
        if line.strip().startswith("✓"):
            ok += 1
        time.sleep(0.4)  # be polite to the API
    print(f"\nDone, {ok}/{len(PLAYERS)} photos saved. The rest keep their generated avatars.")
    print("Restart the app to see the photos on the cards.")


if __name__ == "__main__":
    main()
