"""HLTV scraper: match listings + match pages.

HLTV uses Cloudflare bot protection. We use cloudscraper (JS challenge solver
in Python) and pace requests conservatively. If cloudscraper starts failing,
the fallback is curl-cffi with a real browser TLS fingerprint.

Public surfaces:
    fetch_results(stars=3, offset=0)         -> list[MatchSummary]
    fetch_match(match_id, slug)              -> MatchDetail (with demo URL)
    iter_matches(stars=3, max_matches=100)   -> generator over matches

URL shapes (HLTV is stable, has been for years):
    Listings:  https://www.hltv.org/results?stars={1-5}&offset={0,100,200,...}
    Match:     https://www.hltv.org/matches/{id}/{slug}
    Demo:      https://www.hltv.org/download/demo/{numeric_id}   → returns .rar
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterator

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup

HLTV_BASE = "https://www.hltv.org"
RESULTS_URL = f"{HLTV_BASE}/results"
# curl_cffi impersonate target; chrome124 works well against HLTV's Cloudflare
# (cloudscraper does NOT — Cloudflare's WAF fingerprints the TLS handshake, not
# just JS challenges, and pure-Python TLS gets a 403 "Just a moment...")
DEFAULT_IMPERSONATE = "chrome124"
DEFAULT_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
}
# Be conservative — HLTV bans aggressive scrapers. 4-6s between requests is
# what most polite scrapers settle on.
DEFAULT_REQUEST_DELAY_SEC = 5.0


@dataclass
class MatchSummary:
    """One row from the results page."""
    match_id: int
    slug: str
    team1: str
    team2: str
    event: str
    score: str             # e.g. "2 - 1" or "16 - 14"
    stars: int             # HLTV tier (1=lowest, 5=highest)
    href: str              # full match URL
    date_unix: int | None = None  # parsed from data-zonedgrouping-entry-unix when present


@dataclass
class MatchDetail:
    """Full match metadata + demo URL."""
    summary: MatchSummary
    demo_url: str | None   # https://www.hltv.org/download/demo/<id>
    demo_id: int | None
    maps: list[str] = field(default_factory=list)   # ["mirage", "inferno", ...]


class HLTVScraper:
    def __init__(self, delay_sec: float = DEFAULT_REQUEST_DELAY_SEC,
                 impersonate: str = DEFAULT_IMPERSONATE):
        self.session = cffi_requests.Session(impersonate=impersonate)
        self.session.headers.update(DEFAULT_HEADERS)
        self.delay = delay_sec
        self._last_request_ts = 0.0

    def _get(self, url: str) -> str:
        """GET with delay + retry. Raises on non-200."""
        # Rate limit
        wait = self.delay - (time.time() - self._last_request_ts)
        if wait > 0:
            time.sleep(wait)
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                r = self.session.get(url, timeout=30)
                self._last_request_ts = time.time()
                if r.status_code == 200:
                    return r.text
                if r.status_code in (403, 429, 503):
                    # Cloudflare blocked or rate-limited — back off
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
            except Exception as e:
                last_err = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"HLTV GET failed after retries: {url} ({last_err})")

    # ----- results page ------------------------------------------------------

    def fetch_results(self, stars: int = 3, offset: int = 0) -> list[MatchSummary]:
        """Parse one page of /results. HLTV returns 100 matches/page.

        stars: filter by HLTV match tier (5 = LAN majors, 3 = top online, 1 = all).
        offset: pagination cursor (0, 100, 200, ...).
        """
        url = f"{RESULTS_URL}?stars={stars}&offset={offset}"
        html = self._get(url)
        return self._parse_results(html)

    @staticmethod
    def _parse_results(html: str) -> list[MatchSummary]:
        soup = BeautifulSoup(html, "lxml")
        out: list[MatchSummary] = []
        # Results page: each match is inside <div class="result-con"> with a
        # nested <a class="a-reset"> linking to /matches/{id}/{slug}
        for con in soup.select("div.result-con"):
            a = con.find("a", class_="a-reset")
            if not a or not a.get("href"):
                continue
            href = a["href"]
            m = re.match(r"/matches/(\d+)/([\w\d-]+)", href)
            if not m:
                continue
            match_id = int(m.group(1))
            slug = m.group(2)
            team_cells = a.select("div.team")
            team1 = team_cells[0].get_text(strip=True) if len(team_cells) > 0 else ""
            team2 = team_cells[1].get_text(strip=True) if len(team_cells) > 1 else ""
            score_el = a.select_one("td.result-score") or a.select_one(".result-score")
            score = score_el.get_text(" ", strip=True) if score_el else ""
            event_el = a.select_one("span.event-name") or a.select_one(".event")
            event = event_el.get_text(strip=True) if event_el else ""
            stars_count = len(a.select("i.fa.fa-star.star"))
            unix_el = con.find(attrs={"data-zonedgrouping-entry-unix": True})
            try:
                date_unix = (int(unix_el["data-zonedgrouping-entry-unix"]) // 1000
                             if unix_el else None)
            except Exception:
                date_unix = None
            out.append(MatchSummary(
                match_id=match_id, slug=slug,
                team1=team1, team2=team2, event=event, score=score,
                stars=stars_count,
                href=HLTV_BASE + href,
                date_unix=date_unix,
            ))
        return out

    # ----- match page --------------------------------------------------------

    def fetch_match(self, match_id: int, slug: str) -> MatchDetail:
        """Pull match page and extract demo download URL + map list."""
        url = f"{HLTV_BASE}/matches/{match_id}/{slug}"
        html = self._get(url)
        return self._parse_match(html, match_id, slug)

    def _parse_match(self, html: str, match_id: int, slug: str) -> MatchDetail:
        soup = BeautifulSoup(html, "lxml")
        demo_url: str | None = None
        demo_id: int | None = None
        # HLTV puts GOTV demos in a stripe near the bottom. Anchor:
        #   <a class="stream-box ... " href="/download/demo/<id>" ...>
        # We grab the first /download/demo/ link.
        a = soup.find("a", href=re.compile(r"^/download/demo/\d+"))
        if a:
            demo_url = HLTV_BASE + a["href"]
            m = re.search(r"/download/demo/(\d+)", a["href"])
            demo_id = int(m.group(1)) if m else None

        maps: list[str] = []
        # Map names on map-tabs: <div class="mapname">Mirage</div>
        for el in soup.select("div.mapname"):
            name = el.get_text(strip=True).lower()
            if name and name not in maps and name != "tba":
                maps.append(name)

        # Score header tells us teams (fallback if results parse missed)
        team1 = team2 = ""
        score = ""
        team_els = soup.select("div.teamName") or soup.select(".teamName")
        if len(team_els) >= 2:
            team1 = team_els[0].get_text(strip=True)
            team2 = team_els[1].get_text(strip=True)
        score_el = soup.find(class_=re.compile(r"team\dGradient")) or soup.select_one("div.team1-gradient")
        if score_el:
            sc = score_el.find_all(text=re.compile(r"^\d+$"))
            if len(sc) >= 2:
                score = f"{sc[0].strip()} - {sc[1].strip()}"

        event_el = soup.select_one("div.event a") or soup.select_one(".event-name")
        event = event_el.get_text(strip=True) if event_el else ""

        summary = MatchSummary(
            match_id=match_id, slug=slug,
            team1=team1, team2=team2, event=event, score=score,
            stars=0, href=f"{HLTV_BASE}/matches/{match_id}/{slug}",
        )
        return MatchDetail(summary=summary, demo_url=demo_url, demo_id=demo_id, maps=maps)

    # ----- generator ---------------------------------------------------------

    def iter_matches(self, stars: int = 3, max_matches: int = 100,
                      start_offset: int = 0) -> Iterator[MatchSummary]:
        """Walk paginated results until max_matches yielded or empty page hit."""
        n = 0
        offset = start_offset
        while n < max_matches:
            batch = self.fetch_results(stars=stars, offset=offset)
            if not batch:
                return
            for m in batch:
                yield m
                n += 1
                if n >= max_matches:
                    return
            offset += 100
