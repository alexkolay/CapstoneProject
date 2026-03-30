"""
Fetch active satellite GP data from SpaceTrack for 5+ constellations.

Usage:
    python fetch_constellations.py --username YOUR_EMAIL --password YOUR_PASSWORD

SpaceTrack account registration (free): https://www.space-track.org/auth/createAccount

Output:
    One JSON file per constellation in the current directory, e.g.:
        oneweb_spacetrack.json
    A merged multi-constellation GP table (if you build one) is conventionally
    named spacetrack_data.csv in this project.
        iridium_spacetrack.json
        planet_spacetrack.json
        spire_spacetrack.json
        gps_spacetrack.json
        glonass_spacetrack.json
        globalstar_spacetrack.json
        navstar_spacetrack.json
        orbcomm_spacetrack.json
        spacebee_spacetrack.json
        o3b_spacetrack.json
"""

import argparse
import json
import time
import requests

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Fetch constellation data from SpaceTrack")
parser.add_argument("--username", required=True, help="SpaceTrack account email")
parser.add_argument("--password", required=True, help="SpaceTrack account password")
args = parser.parse_args()

BASE_URL   = "https://www.space-track.org"
LOGIN_URL  = f"{BASE_URL}/ajaxauth/login"
LOGOUT_URL = f"{BASE_URL}/ajaxauth/logout"
GP_BASE    = f"{BASE_URL}/basicspacedata/query/class/gp"

# ── Constellations to fetch ───────────────────────────────────────────────────
# Each entry: (output filename, OBJECT_NAME filter)
# SpaceTrack uses ~~ for "contains" matching on OBJECT_NAME
CONSTELLATIONS = [
    ("oneweb_spacetrack.json",      "ONEWEB"),
    ("iridium_spacetrack.json",     "IRIDIUM"),
    ("planet_spacetrack.json",      "FLOCK"),       # Planet Labs Dove/Flock satellites
    ("spire_spacetrack.json",       "LEMUR"),        # Spire Global nanosats
    ("glonass_spacetrack.json",     "COSMOS"),       # GLONASS satellites (Russian Cosmos series)
    ("globalstar_spacetrack.json",  "GLOBALSTAR"),   # Globalstar LEO voice/data
    ("navstar_spacetrack.json",     "NAVSTAR"),      # GPS (correct SpaceTrack name)
    ("orbcomm_spacetrack.json",     "ORBCOMM"),      # ORBCOMM LEO IoT
    ("spacebee_spacetrack.json",    "SPACEBEE"),     # Swarm Technologies nanosats
    ("o3b_spacetrack.json",         "O3B"),          # SES O3B MEO broadband
]

def build_query(name_filter):
    """
    Build SpaceTrack GP query URL for active satellites matching a name filter.
    DECAY_DATE/null-val restricts to active (non-decayed) satellites only.
    OBJECT_TYPE/PAYLOAD excludes rocket bodies and debris.
    """
    return (
        f"{GP_BASE}"
        f"/OBJECT_NAME/~~{name_filter}"
        f"/OBJECT_TYPE/PAYLOAD"
        f"/DECAY_DATE/null-val"
        f"/orderby/NORAD_CAT_ID"
        f"/format/json"
    )

# ── Login ─────────────────────────────────────────────────────────────────────
session = requests.Session()
print("Logging in to SpaceTrack...")
resp = session.post(LOGIN_URL, data={"identity": args.username, "password": args.password})
resp.raise_for_status()
if "Invalid" in resp.text or "incorrect" in resp.text.lower():
    raise RuntimeError("SpaceTrack login failed — check your username/password.")
print("Login successful.\n")

# ── Fetch each constellation ──────────────────────────────────────────────────
for filename, name_filter in CONSTELLATIONS:
    url = build_query(name_filter)
    print(f"Fetching {name_filter} satellites → {filename} ...")

    resp = session.get(url)
    resp.raise_for_status()
    data = resp.json()

    if not data:
        print(f"  WARNING: No records returned for {name_filter}. Skipping.\n")
        continue

    with open(filename, "w") as f:
        json.dump(data, f)

    print(f"  Saved {len(data)} active satellites → {filename}\n")

    # SpaceTrack rate limit: max 20 requests per minute
    time.sleep(3)

# ── Logout ────────────────────────────────────────────────────────────────────
session.get(LOGOUT_URL)
print("Logged out. All done.")
