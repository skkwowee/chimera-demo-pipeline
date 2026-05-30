"""VOD discovery + demo-to-VOD matching.

Goal: for each match in the demo manifest, find the YouTube VOD(s) for each
map, so we can later pull caster commentary and align it to demo ticks (the
"ground verbalization in real commentary" plan).

Why this is tractable rather than fuzzy guesswork:
  - Our demo files are per-map and self-label the map:
        spirit-vs-falcons-m1-dust2.dem  ->  (map_index=1, map=dust2)
  - Official tournament VODs ALSO self-label, in the title:
        "PGL Astana 2026 - Spirit vs. Falcons (Mapa 1 - Dust 2) - 17/05/2026"
  So matching is: search by "{team1} vs {team2} {event}", then parse each
  candidate's title for (teams, event, map name, map index) and score the
  overlap against the demo metadata. A correct per-map VOD scores high on
  ALL of teams + event + map; a highlight reel or wrong-match VOD does not.

This module does DISCOVERY + SCORING only (pure metadata, no downloads).
Audio extraction + ASR + tick alignment live in later modules (they need
ffmpeg / whisper, which aren't installed yet).

Public API:
    parse_demo_file(path)            -> (map_index, map_name)
    score_candidate(cand, match)     -> MatchScore
    find_vods_for_match(match, ...)  -> list[VodMatch]   (best per map)
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field

# Canonical CS2 active-duty + common map names, with title aliases.
# Keys are canonical (as they appear in our demo filenames); values are
# alternate spellings that show up in VOD titles ("Dust 2", "d2", etc.).
MAP_ALIASES = {
    "dust2": ["dust2", "dust 2", "dust ii", "d2"],
    "mirage": ["mirage"],
    "inferno": ["inferno"],
    "nuke": ["nuke"],
    "ancient": ["ancient"],
    "anubis": ["anubis"],
    "overpass": ["overpass"],
    "vertigo": ["vertigo"],
    "train": ["train"],
}

# "map 1", "mapa 1", "m1", "map1", "карта 1" (RU) -> index
_MAP_IDX_RE = re.compile(
    r"\b(?:map|mapa|mapě|mappa|karte|карта|m)\s*\.?\s*(\d)\b", re.IGNORECASE)


@dataclass
class VodCandidate:
    video_id: str
    title: str
    duration: int | None
    channel: str | None = None


@dataclass
class MatchScore:
    total: float
    team1_hit: bool
    team2_hit: bool
    event_hit: bool
    map_hit: bool
    map_index_hit: bool
    is_highlight: bool          # penalize "highlights"/"top plays" reels
    reasons: list[str] = field(default_factory=list)


@dataclass
class VodMatch:
    map_index: int
    map_name: str
    demo_file: str
    candidate: VodCandidate | None
    score: MatchScore | None


# ---------------------------------------------------------------------------
# demo metadata parsing
# ---------------------------------------------------------------------------
def parse_demo_file(path: str) -> tuple[int | None, str | None]:
    """`demos/spirit-vs-falcons-m1-dust2.dem` -> (1, 'dust2')."""
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    midx = None
    m = re.search(r"-m(\d)-", stem)
    if m:
        midx = int(m.group(1))
    mapn = None
    for canon in MAP_ALIASES:
        if re.search(rf"(?:^|-){canon}(?:-|$)", stem):
            mapn = canon
            break
    return midx, mapn


def _norm_team(name: str) -> list[str]:
    """Team name -> searchable tokens. Handles 'Natus Vincere'->'navi' etc."""
    n = name.lower().strip()
    aliases = {
        "natus vincere": ["natus vincere", "navi", "na'vi", "na vi"],
        "the mongolz": ["mongolz", "the mongolz"],
        "g2": ["g2"],
        "faze": ["faze"],
    }
    if n in aliases:
        return aliases[n]
    return [n]


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
HIGHLIGHT_MARKERS = ["highlight", "highlights", "top play", "top plays",
                     "best of", "montage", "frags", "fragmovie", "review",
                     "analysis", "recap"]

# Weights — teams + map are the load-bearing signals; event + index refine.
W_TEAM = 1.0       # each team
W_EVENT = 1.0
W_MAP = 2.0        # the map name is the strongest per-map disambiguator
W_MAP_INDEX = 1.0
HIGHLIGHT_PENALTY = -3.0


def score_candidate(cand: VodCandidate, team1: str, team2: str, event: str,
                    map_name: str | None, map_index: int | None) -> MatchScore:
    title = cand.title.lower()
    reasons: list[str] = []

    t1_hit = any(tok in title for tok in _norm_team(team1))
    t2_hit = any(tok in title for tok in _norm_team(team2))

    # Event: match on distinctive event tokens (drop year-agnostic filler).
    ev_tokens = [t for t in re.split(r"\s+", event.lower())
                 if len(t) > 2 and t not in ("the", "cs2", "season")]
    ev_hits = sum(1 for t in ev_tokens if t in title)
    event_hit = ev_hits >= max(1, len(ev_tokens) // 2)

    map_hit = False
    if map_name:
        map_hit = any(alias in title for alias in MAP_ALIASES.get(map_name, [map_name]))

    map_index_hit = False
    if map_index is not None:
        m = _MAP_IDX_RE.search(cand.title)
        if m and int(m.group(1)) == map_index:
            map_index_hit = True

    is_hl = any(mk in title for mk in HIGHLIGHT_MARKERS)

    total = 0.0
    if t1_hit: total += W_TEAM; reasons.append("team1")
    if t2_hit: total += W_TEAM; reasons.append("team2")
    if event_hit: total += W_EVENT; reasons.append("event")
    if map_hit: total += W_MAP; reasons.append(f"map:{map_name}")
    if map_index_hit: total += W_MAP_INDEX; reasons.append(f"m{map_index}")
    if is_hl: total += HIGHLIGHT_PENALTY; reasons.append("HIGHLIGHT_PENALTY")

    return MatchScore(
        total=total, team1_hit=t1_hit, team2_hit=t2_hit, event_hit=event_hit,
        map_hit=map_hit, map_index_hit=map_index_hit, is_highlight=is_hl,
        reasons=reasons,
    )


# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------
def yt_search(query: str, n: int = 8) -> list[VodCandidate]:
    """Search YouTube via yt-dlp (metadata only, no download).

    NOTE: we deliberately do NOT use --flat-playlist. Flat mode is faster but
    returns TRUNCATED titles for some entries (cutting off the trailing
    "...2026" date), which silently drops event/team token matches and tanks
    the score. Full extraction (`--skip-download`) returns complete titles +
    real durations, at the cost of a few extra seconds per search.
    """
    cmd = ["yt-dlp", "--skip-download", "--no-warnings", "--dump-json",
           "--ignore-errors", f"ytsearch{n}:{query}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return []
    seen: set[str] = set()
    cands: list[VodCandidate] = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        vid = d.get("id")
        if not vid or vid in seen:
            continue
        seen.add(vid)
        cands.append(VodCandidate(
            video_id=vid, title=d.get("title", "") or "",
            duration=d.get("duration"), channel=d.get("channel"),
        ))
    return cands


# A demo (a single map) typically runs 25-55 min; full-series broadcasts run
# 1.5-5h. We accept both but flag obvious mismatches (a per-map VOD should
# be in the half-hour-to-hour range).
MIN_MAP_DURATION = 12 * 60       # 12 min — shorter is a highlight/clip
MAX_MAP_DURATION = 90 * 60       # 90 min — longer is likely a full series


def _best_over_pool(pool, team1, team2, event, mapn, midx):
    best_c, best_s = None, None
    for c in pool:
        s = score_candidate(c, team1, team2, event, mapn, midx)
        if c.duration is not None and c.duration < MIN_MAP_DURATION:
            s.total += HIGHLIGHT_PENALTY
            s.reasons.append("too_short")
        if best_s is None or s.total > best_s.total:
            best_c, best_s = c, s
    return best_c, best_s


def find_vods_for_match(match: dict, n_results: int = 8,
                        min_score: float = 3.0) -> list[VodMatch]:
    """For each map in the match, search + score + pick the best VOD.

    Two-pass discovery per map:
      1. Generic series search ("{t1} vs {t2} {event}") — shared pool, cheap.
      2. If no map clears min_score from the shared pool, do a TARGETED
         per-map search that injects the map name + index into the query
         ("{t1} vs {t2} {event} map {i} {mapname}"). Official broadcasts
         upload one VOD per map, so the targeted query surfaces the right
         one even when the generic top-N only had map 1.

    min_score default 3.0: map name (W_MAP=2) + one team, or both teams +
    event. A correct per-map VOD scores ~6 (both teams + event + map + idx).
    """
    team1, team2, event = match["team1"], match["team2"], match["event"]
    shared_pool = yt_search(f"{team1} vs {team2} {event}", n=n_results)

    out: list[VodMatch] = []
    for demo_file in match.get("demo_files", []):
        midx, mapn = parse_demo_file(demo_file)
        best_c, best_s = _best_over_pool(shared_pool, team1, team2, event, mapn, midx)
        # Targeted per-map fallback when the shared pool didn't yield a winner
        if best_s is None or best_s.total < min_score:
            q = f"{team1} vs {team2} {event} map {midx} {mapn or ''}".strip()
            tc, ts = _best_over_pool(yt_search(q, n=n_results),
                                     team1, team2, event, mapn, midx)
            if ts is not None and (best_s is None or ts.total > best_s.total):
                best_c, best_s = tc, ts
        if best_s is None or best_s.total < min_score:
            best_c = None
        out.append(VodMatch(
            map_index=midx or 0, map_name=mapn or "?",
            demo_file=demo_file, candidate=best_c, score=best_s,
        ))
    return out
