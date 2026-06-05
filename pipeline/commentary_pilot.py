"""Commentary-grounding pilot: is caster speech a usable supervision source?

This is the cheap, falsify-it-in-an-afternoon slice described in
docs/commentary-grounding.md. It answers ONE question before we invest in a
training rig: when you strip the off-topic chatter, is there enough *specific
tactical* commentary, and does it land on the windows where the structured
features are blind (the "structurally-flat" windows)?

Three failure modes of caster speech, by severity:
  1. SILENCE        — benign. A window with no speech is an explicit null /
                      negative, not a missing label. Costs coverage, not validity.
  2. OFF-TOPIC      — tractable. "nice crowd", "remember to subscribe". Removable
                      with a lexical relevance gate (no LLM needed -> not circular).
  3. VAGUE-ON-TOPIC — the real danger. "they're being patient", "good util".
                      About the game but too low-res to supervise fine structure.
                      A keyword gate CANNOT catch paraphrase ("stacking that
                      corner"), so the gate bounds RECALL: we therefore emit the
                      flat bucket for hand-reading REGARDLESS of gate verdict, so
                      we measure the gate's recall rather than trusting it.

Alignment note (the lag has a SIGN): reaction commentary LAGS events ("and
Spirit take it"); anticipatory tactical reads LEAD them ("watch the fake here").
A global offset fit on kill/round anchors fits the *reaction* lag and pushes
forward-looking calls into the prior window. So the demo<->VOD window search is
ASYMMETRIC (look earlier as well as later) -- see LEAD_SEC / LAG_SEC.

Runs in two halves:
  - caption half (this file, no heavy deps): VTT parse + window + relevance gate.
    RUNS ANYWHERE yt-dlp can fetch auto-subs (no ffmpeg, no GPU).
  - alignment half (needs demo anchors from awpy on the pod): fit offset, attach
    structural_flatness, bucket. Anchor/flatness fns are pure given their inputs.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# VTT parsing (YouTube auto-subs are a rolling-window format: each cue repeats
# the previous line then adds new words with inline <ts><c> word timing. We
# pull the per-WORD timestamps and dedup, which gives clean word-level timing.)
# ---------------------------------------------------------------------------
@dataclass
class Word:
    t: float          # start time in VOD seconds
    text: str


_TS_TAG = re.compile(r"<(\d{2}):(\d{2}):(\d{2})\.(\d{3})>")
_C_TAG = re.compile(r"</?c[^>]*>")
_CUE_TIME = re.compile(
    r"(\d{2}):(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})\.(\d{3})")


def _hms(h, m, s, ms) -> float:
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_vtt(path: str) -> list[Word]:
    """Parse a YouTube auto-sub .vtt into a deduped, time-ordered word list."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    words: list[Word] = []
    cur_cue_start = 0.0
    for block in raw.split("\n\n"):
        lines = block.strip().splitlines()
        if not lines:
            continue
        for ln in lines:
            mt = _CUE_TIME.search(ln)
            if mt:
                cur_cue_start = _hms(*mt.groups()[:4])
                continue
            if "<" not in ln and "-->" not in ln:
                continue  # plain rolled-up duplicate line, skip
            # Inline-timed line: split on the <ts> tags to recover word times.
            # First word inherits the cue start; each subsequent <ts> stamps the
            # word that follows it.
            parts = _TS_TAG.split(ln)
            # parts = [text0, h,m,s,ms, text1, h,m,s,ms, text2, ...]
            seg0 = _C_TAG.sub("", parts[0]).strip()
            if seg0:
                for w in seg0.split():
                    words.append(Word(cur_cue_start, w))
            i = 1
            while i + 4 < len(parts) + 1 and i + 3 < len(parts):
                t = _hms(parts[i], parts[i + 1], parts[i + 2], parts[i + 3])
                seg = _C_TAG.sub("", parts[i + 4]).strip() if i + 4 < len(parts) else ""
                for w in seg.split():
                    words.append(Word(t, w))
                i += 5
    # Dedup: the rolling format re-emits the same (t, word); keep first occurrence.
    seen: set[tuple] = set()
    out: list[Word] = []
    for w in words:
        clean = html.unescape(w.text)
        key = (round(w.t, 2), clean.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(Word(w.t, clean))
    out.sort(key=lambda w: w.t)
    return out


# ---------------------------------------------------------------------------
# the relevance gate (lexical; NO model -> avoids circularity with Claude caps)
# ---------------------------------------------------------------------------
# Generic callouts that recur across maps + the dust2-specific set (the pilot
# VOD is a dust2 map). Add per-map sets as we extend to other maps.
CALLOUTS = {
    # generic
    "mid", "long", "short", "site", "ramp", "connector", "jungle", "catwalk",
    "palace", "apartments", "apps", "banana", "pit", "heaven", "hell", "window",
    "ninja", "spawn", "tunnels", "tunnel", "upper", "lower", "main", "stairs",
    "boost", "default", "back", "front", "cross", "doors", "elevator",
    # dust2
    "car", "goose", "barrels", "xbox", "suicide", "tetris", "ct", "tspawn",
}
TACTICS = {
    "fake", "faking", "rotate", "rotating", "rotation", "rotations", "execute",
    "executing", "lurk", "lurking", "lurker", "stack", "stacked", "stacking",
    "split", "splitting", "retake", "retaking", "save", "saving", "eco",
    "force", "forcing", "flank", "flanking", "peek", "peeking", "hold",
    "holding", "trade", "trading", "refrag", "crossfire", "pick", "picks",
    "info", "control", "post-plant", "postplant", "plant", "planted",
    "defuse", "defusing", "clutch", "clutching", "entry", "support", "anchor",
    "aggression", "aggressive", "passive", "contact", "commit", "commitment",
    "read", "anti-eco", "antieco", "double", "pinch", "wrap", "off-angle",
    "offangle", "isolate", "duel", "advantage", "manadvantage",
}
UTILITY = {
    "smoke", "smokes", "smoking", "flash", "flashes", "flashed", "flashbang",
    "molotov", "molly", "incendiary", "incen", "nade", "nades", "grenade",
    "grenades", "popflash", "one-way", "oneway", "wallbang", "util",
    "utility", "lineup", "lineups",
    # NOTE: "he" (HE grenade) is DELIBERATELY EXCLUDED -- it collides with the
    # English pronoun and produced 205 false-positive hits on the pilot VOD,
    # inflating the tactical bucket 49%->39%. Homographs must not be in the
    # high-precision set; see HOMOGRAPHS below.
}
# Tactical terms that are ALSO common English words. They DO carry signal
# ("long"/"short"/"back" are real dust2 callouts; "pick"/"trade"/"hold" are
# real verbs) but a bare match is untrustworthy. Counted only as a WEAK hit:
# valid only when corroborated by a high-precision token in the same window.
HOMOGRAPHS = {
    "long", "short", "back", "front", "cross", "main", "double", "pick",
    "picks", "trade", "trading", "hold", "holding", "read", "contact", "save",
    "saving", "control", "support", "advantage", "default", "window", "site",
    "mid", "ct", "info", "boost", "stairs", "ramp", "car",
}
ECON = {
    "buy", "buying", "fullbuy", "full-buy", "forcebuy", "force-buy", "money",
    "economy", "loss", "bonus", "broke", "rifles", "armor", "kevlar",
}
WEAPONS = {
    "awp", "ak", "ak47", "m4", "deagle", "scout", "ssg", "aug", "krieg",
    "sg", "galil", "famas", "mac", "mp9", "usp", "glock", "tec",
}
# emotive / filler that signals on-topic-but-vague (NOT specific)
VAGUE = {
    "nice", "good", "great", "huge", "massive", "incredible", "amazing",
    "beautiful", "perfect", "insane", "crazy", "unreal", "wow", "patient",
    "patiently", "slow", "slowly", "well", "clean", "smart", "big", "lovely",
}
# clear off-topic / broadcast-furniture markers
OFFTOPIC = {
    "subscribe", "crowd", "sponsor", "sponsored", "drink", "twitter", "tweet",
    "instagram", "channel", "break", "desk", "analyst", "studio", "viewers",
    "chat", "merch", "giveaway", "welcome", "betting", "odds", "prediction",
}

# CALLOUTS still contains some homographs (mid/site/long...) for readability;
# the high-precision SPECIFIC set excludes anything in HOMOGRAPHS so a
# "specific" hit is trustworthy. Homographs are scored separately (weak).
SPECIFIC = (CALLOUTS | TACTICS | UTILITY | ECON | WEAPONS) - HOMOGRAPHS


@dataclass
class Window:
    t_start: float
    t_end: float
    text: str
    n_words: int
    specific_hits: list[str] = field(default_factory=list)
    vague_hits: list[str] = field(default_factory=list)
    offtopic_hits: list[str] = field(default_factory=list)
    bucket: str = ""          # tactical | vague | offtopic | silence
    # filled in by the alignment half (pod):
    demo_round: int | None = None
    demo_tick_range: tuple[int, int] | None = None
    structurally_flat: bool | None = None

    def to_dict(self) -> dict:
        return {
            "t_start": round(self.t_start, 1),
            "t_end": round(self.t_end, 1),
            "bucket": self.bucket,
            "n_words": self.n_words,
            "n_specific": len(self.specific_hits),
            "specific_hits": self.specific_hits,
            "vague_hits": self.vague_hits,
            "offtopic_hits": self.offtopic_hits,
            "demo_round": self.demo_round,
            "demo_tick_range": self.demo_tick_range,
            "structurally_flat": self.structurally_flat,
            "text": self.text,
        }


_WORD = re.compile(r"[a-z][a-z'\-]*")


def gate_text(text: str) -> tuple[str, list[str], list[str], list[str]]:
    """Classify a window's text. Returns (bucket, specific, vague, offtopic).

    bucket: 'tactical' if it names specific tactical content (>=2 specific
    tokens, OR >=1 callout-or-utility + >=1 tactic verb -> a real call);
    'offtopic' if broadcast-furniture markers dominate and no specific tokens;
    'vague' if on-topic-emotive but not specific; 'silence' if ~no words.
    """
    toks = _WORD.findall(text.lower())
    if len(toks) < 3:
        return "silence", [], [], []
    spec = [t for t in toks if t in SPECIFIC]            # high-precision
    weak = [t for t in toks if t in HOMOGRAPHS]          # only count if corroborated
    vague = [t for t in toks if t in VAGUE]
    offt = [t for t in toks if t in OFFTOPIC]
    # A homograph counts toward specificity ONLY when a high-precision token
    # also fired in the window (e.g. "smoke long" -> "long" is a real callout;
    # "looked back" with no other CS token -> "back" stays noise).
    corroborated = spec + (weak if spec else [])
    distinct = set(corroborated)
    has_call = any(t in (CALLOUTS | UTILITY) and t not in HOMOGRAPHS for t in spec)
    has_tac = any(t in TACTICS for t in spec)
    is_specific = len(distinct) >= 2 or (has_call and has_tac)
    spec = corroborated
    if is_specific:
        return "tactical", spec, vague, offt
    if offt and not spec:
        return "offtopic", spec, vague, offt
    if vague and not spec:
        return "vague", spec, vague, offt
    if spec:           # exactly one specific token, no second signal
        return "vague", spec, vague, offt
    return "offtopic", spec, vague, offt


def window_words(words: list[Word], win_sec: float = 15.0,
                 hop_sec: float | None = None) -> list[Window]:
    """Tile words into fixed windows and gate each."""
    if not words:
        return []
    hop = hop_sec or win_sec
    t0, tN = words[0].t, words[-1].t
    out: list[Window] = []
    t = t0
    wi = 0
    while t <= tN:
        lo, hi = t, t + win_sec
        chunk = [w for w in words if lo <= w.t < hi]
        text = " ".join(w.text for w in chunk)
        bucket, spec, vague, offt = gate_text(text)
        out.append(Window(
            t_start=lo, t_end=hi, text=text, n_words=len(chunk),
            specific_hits=spec, vague_hits=vague, offtopic_hits=offt,
            bucket=bucket,
        ))
        t += hop
        wi += 1
    return out


# ---------------------------------------------------------------------------
# alignment half (pure given inputs; the awpy demo-anchor fetch is the only
# part that needs the pod -- it lives in the runner, not here).
# ---------------------------------------------------------------------------
LEAD_SEC = 8.0      # anticipatory calls precede the event by up to this much
LAG_SEC = 4.0       # reaction calls trail it by up to this much


def fit_global_offset(demo_anchor_t: list[float], vod_anchor_t: list[float],
                      max_lag: float = 600.0, step: float = 0.5,
                      tol: float = 2.0, min_lag: float | None = None,
                      ) -> tuple[float, float]:
    """Recover VOD->demo offset by cross-correlating two event trains.

    demo_anchor_t = demo-time stamps of one landmark type (kills). vod_anchor_t =
    VOD-time stamps of the matching landmark (name mentions). A kill at demo time
    t maps to VOD time t+offset; we slide offset and count how many kills have AT
    LEAST ONE name within `tol` (coverage), NOT a greedy 1:1 match — dense names
    near one kill must not consume anchors for the next. Returns (offset, score)
    where demo_t ~= vod_t - offset. Pure / unit-testable; no deps.

    Note on signal: with lossy auto-caption ASR, peak coverage may be only
    ~2-4x the chance floor — still a statistically sharp offset (verify the peak
    is several sigma over the off-peak mean), but per-kill anchoring is sparse.
    """
    import bisect
    if not demo_anchor_t or not vod_anchor_t:
        return 0.0, 0.0
    names = sorted(vod_anchor_t)
    d = sorted(demo_anchor_t)
    lo = -max_lag if min_lag is None else min_lag
    best_off, best_score = lo, -1.0
    lag = lo
    while lag <= max_lag:
        score = 0
        for t in d:
            x = t + lag
            k = bisect.bisect_left(names, x - tol)
            if k < len(names) and names[k] <= x + tol:
                score += 1
        if score > best_score:
            best_score, best_off = score, lag
        lag += step
    return float(best_off), float(best_score)


def vod_window_for_event(event_demo_t: float, offset: float) -> tuple[float, float]:
    """Asymmetric VOD window for a demo event: look EARLIER (anticipation) and
    a little later (reaction). This is the lag-sign fix."""
    center = event_demo_t + offset
    return center - LEAD_SEC, center + LAG_SEC


# ---------------------------------------------------------------------------
# transcript-side anchors: player-name mentions (the densest landmark — every
# kill is a name the demo knows exactly).
# ---------------------------------------------------------------------------
_LEET = str.maketrans({"1": "i", "0": "o", "3": "e", "4": "a", "5": "s", "7": "t"})


def normalize_nick(s: str) -> str:
    """Fold a nickname for fuzzy matching: lowercase, de-leet (sh1ro->shiro,
    zont1x->zontix), strip non-alpha. ASR writes names phonetically so demo
    'donk'/'sh1ro' must match transcript 'donk'/'shiro'."""
    return "".join(c for c in s.lower().translate(_LEET) if c.isalpha())


def phonetic_skeleton(s: str) -> str:
    """Consonant skeleton with phonetic folds — collapses the many ASR
    spellings of a nickname to one key. ASR writes 'tenir'/'tanier' for the
    demo handle 'tN1R'; both skeleton to 'tnr'. Validated to recover tN1R,
    m0NESY, kyousuke, kyxsan, chopper, donk, zweih with no common-word hits."""
    z = normalize_nick(s)
    for a, b in (("ph", "f"), ("ck", "k"), ("x", "ks"), ("ch", "k"),
                 ("sh", "s"), ("c", "k"), ("v", "w"), ("q", "k")):
        z = z.replace(a, b)
    out: list[str] = []
    for c in z:
        if c in "aeiouy":
            continue
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)


def build_name_vocab(roster: list[str]) -> tuple[dict[str, str], dict[str, str]]:
    """roster -> (exact_vocab, skeleton_vocab).

    exact_vocab keys on the de-leeted name (>=3 chars) — handles names whose
    skeleton is too short/ambiguous (NiKo->nk, TeSeS->ts, sh1ro->sr).
    skeleton_vocab keys on the phonetic skeleton (>=3 chars only — shorter keys
    collide with English, e.g. 'ts'=="it's"). A token matches if EITHER fires.
    """
    exact: dict[str, str] = {}
    skel: dict[str, str] = {}
    for name in roster:
        n = normalize_nick(name)
        if len(n) >= 3:
            exact[n] = name
        sk = phonetic_skeleton(name)
        if len(sk) >= 3:
            skel[sk] = name
    return exact, skel


def find_name_mentions(words: list[Word], roster: list[str]) -> list[float]:
    """Times (VOD sec) at which a roster name is spoken (exact OR phonetic)."""
    exact, skel = build_name_vocab(roster)
    out: list[float] = []
    for w in words:
        n = normalize_nick(w.text)
        if not n:
            continue
        if n in exact or phonetic_skeleton(w.text) in skel:
            out.append(w.t)
    return out


# ---------------------------------------------------------------------------
# end-to-end alignment: cross-correlate, then map windows to rounds/ticks.
# ---------------------------------------------------------------------------
@dataclass
class Alignment:
    offset: float          # demo_sec ~= vod_sec - offset
    score: float           # coincidence count at best offset
    n_demo: int
    n_vod: int


def align_transcript(words: list[Word], kill_secs: list[float],
                     roster: list[str], max_lag: float = 1800.0) -> Alignment:
    """Recover the VOD->demo offset by cross-correlating kill times against
    name-mention times. A correct pairing yields a sharp, high coincidence
    score; a wrong VOD scores near chance (this doubles as match verification)."""
    name_t = find_name_mentions(words, roster)
    offset, score = fit_global_offset(kill_secs, name_t, max_lag=max_lag)
    return Alignment(offset=offset, score=score,
                     n_demo=len(kill_secs), n_vod=len(name_t))


def is_structurally_flat(tick_start: int, tick_end: int,
                         kill_ticks: list[int],
                         bomb_ticks: list[int]) -> bool:
    """A window is 'flat' (features ~uninformative -> where exogenous commentary
    would matter) when nothing discrete happens in it: no kill, no bomb plant."""
    for kt in kill_ticks:
        if tick_start <= kt < tick_end:
            return False
    for bt in bomb_ticks:
        if tick_start <= bt < tick_end:
            return False
    return True


def align_windows(windows: list[Window], alignment: Alignment, anchors,
                  ) -> list[Window]:
    """Attach demo_round / demo_tick_range / structurally_flat to each window
    via the recovered offset. Windows outside any round are dropped."""
    tr = anchors.tickrate
    kill_ticks = [k.tick for k in anchors.kills]
    bomb_ticks = [r.bomb_plant for r in anchors.rounds if r.bomb_plant]
    out: list[Window] = []
    for w in windows:
        demo_s0 = w.t_start - alignment.offset
        demo_s1 = w.t_end - alignment.offset
        t0, t1 = int(demo_s0 * tr), int(demo_s1 * tr)
        rnd = anchors.round_for_tick(t0)
        if rnd is None:
            continue
        w.demo_round = rnd
        w.demo_tick_range = (t0, t1)
        w.structurally_flat = is_structurally_flat(t0, t1, kill_ticks, bomb_ticks)
        out.append(w)
    return out
