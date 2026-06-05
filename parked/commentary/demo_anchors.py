"""Demo anchors for commentary alignment (awpy 2.x).

The demo gives the exact wall-clock sequence of kills and round transitions,
with player names. Those are the landmarks we cross-correlate against the ASR
transcript to recover the VOD->demo time offset (see commentary_pilot.align_*).

awpy 2.0.2 schema (verified on spirit-vs-falcons-m2-dust2):
  d.tickrate -> 128
  d.header   -> dict incl. 'map_name', 'server_name' (carries the EVENT, e.g.
                "StarLadder Major Budapest 2025 | A")
  d.rounds   -> round_num, start, freeze_end, end, official_end, winner, reason,
                bomb_plant (tick|None), bomb_site
  d.kills    -> tick, round_num, attacker_name, victim_name, weapon, headshot,
                attacker_X/Y, victim_X/Y, ... (ticks are ABSOLUTE)
  d.ticks    -> health, place, side, X, Y, Z, tick, name, round_num
                (needs player_props at parse time)

Anchors need NO player props; frame extraction needs X/Y/health.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class KillAnchor:
    tick: int
    sec: float
    round_num: int
    attacker: str
    victim: str
    weapon: str
    headshot: bool


@dataclass
class RoundAnchor:
    round_num: int
    start: int
    freeze_end: int
    end: int
    official_end: int
    winner: str
    reason: str
    bomb_plant: int | None
    bomb_site: str | None

    def sec(self, tickrate: int) -> tuple[float, float]:
        return self.start / tickrate, self.end / tickrate


@dataclass
class DemoAnchors:
    stem: str
    map_name: str
    event: str
    tickrate: int
    rounds: list[RoundAnchor]
    kills: list[KillAnchor]
    roster: list[str] = field(default_factory=list)

    def kill_secs(self) -> list[float]:
        return [k.sec for k in self.kills]

    def round_for_tick(self, tick: int) -> int | None:
        for r in self.rounds:
            if r.start <= tick < r.official_end:
                return r.round_num
        return None


def _col(df, name, default=None):
    """Polars column -> python list, tolerant of missing columns."""
    try:
        return df[name].to_list()
    except Exception:
        return None


def parse_demo(path: str, with_frames: bool = True):
    """Parse a demo. Returns (DemoAnchors, demo_obj). demo_obj is kept so the
    caller can pull per-tick frames (demo_obj.ticks) without reparsing."""
    from awpy import Demo
    d = Demo(path)
    d.parse(player_props=["X", "Y", "Z", "health"] if with_frames else [])
    tickrate = int(getattr(d, "tickrate", 128) or 128)
    header = d.header if isinstance(d.header, dict) else {}
    map_name = header.get("map_name", "")
    event = header.get("server_name", "")
    stem = path.rsplit("/", 1)[-1].rsplit(".", 1)[0]

    rk = d.rounds.to_dicts()
    rounds = [RoundAnchor(
        round_num=r["round_num"], start=r["start"], freeze_end=r["freeze_end"],
        end=r["end"], official_end=r.get("official_end", r["end"]),
        winner=r.get("winner", ""), reason=r.get("reason", ""),
        bomb_plant=r.get("bomb_plant"), bomb_site=r.get("bomb_site"),
    ) for r in rk]

    kk = d.kills.to_dicts()
    kills, roster = [], set()
    for k in kk:
        atk = k.get("attacker_name") or ""
        vic = k.get("victim_name") or ""
        if atk and atk != "world":
            roster.add(atk)
        if vic:
            roster.add(vic)
        kills.append(KillAnchor(
            tick=int(k["tick"]), sec=int(k["tick"]) / tickrate,
            round_num=int(k.get("round_num", 0)),
            attacker=atk, victim=vic, weapon=k.get("weapon", ""),
            headshot=bool(k.get("headshot", False)),
        ))

    anchors = DemoAnchors(
        stem=stem, map_name=map_name, event=event, tickrate=tickrate,
        rounds=rounds, kills=kills, roster=sorted(roster),
    )
    return anchors, d


def extract_round_frames(demo_obj, round_num: int, anchors: DemoAnchors,
                         step: int = 64) -> dict:
    """Pull real per-tick player frames for one round from demo_obj.ticks,
    downsampled to every `step` ticks. Shape matches the viewer's round_NN.json.
    """
    r = next((x for x in anchors.rounds if x.round_num == round_num), None)
    if r is None:
        raise ValueError(f"round {round_num} not found")
    ticks = demo_obj.ticks
    sub = ticks.filter(
        (ticks["round_num"] == round_num)
        & (ticks["tick"] >= r.start) & (ticks["tick"] <= r.official_end)
    )
    rows = sub.to_dicts()
    by_tick: dict[int, list] = {}
    for row in rows:
        by_tick.setdefault(int(row["tick"]), []).append(row)
    frames = []
    for t in sorted(by_tick):
        if (t - r.start) % step != 0:
            continue
        players = []
        for p in by_tick[t]:
            hp = p.get("health") or 0
            # NOTE: the viewer's PlayerFrame uses LOWERCASE x/y/z (z drives the
            # multi-level radar). awpy ticks expose uppercase X/Y/Z.
            players.append({
                "name": p.get("name", ""),
                "side": (p.get("side") or "").upper(),
                "alive": hp > 0,
                "hp": int(hp),
                "x": round(float(p["X"]), 1) if p.get("X") is not None else 0.0,
                "y": round(float(p["Y"]), 1) if p.get("Y") is not None else 0.0,
                "z": round(float(p["Z"]), 1) if p.get("Z") is not None else 0.0,
                "yaw": 0.0,
            })
        frames.append({"tick": t, "players": players})
    return {
        "round_num": round_num, "start_tick": r.start, "end_tick": r.official_end,
        "freeze_end": r.freeze_end, "winner": r.winner, "reason": r.reason,
        "bomb_plant_tick": r.bomb_plant, "bomb_site": r.bomb_site,
        "frames": frames,
    }
