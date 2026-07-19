"""Selector fixture tests: recorded-shape HTML guards HLTV layout drift.

If HLTV changes markup, these fixtures keep passing (they encode the OLD
shape) while the live scraper raises ScrapeLayoutError — which is exactly
the contract: drift must fail loudly, never parse as 'no matches'.
"""

from pathlib import Path

import pytest

from pipeline.hltv import HLTVScraper, ScrapeLayoutError

FIXTURES = Path(__file__).parent / "fixtures"


def read(name: str) -> str:
    return (FIXTURES / name).read_text()


def make_scraper(html: str) -> HLTVScraper:
    s = HLTVScraper(delay_sec=0)
    s._get = lambda url: html   # no network
    return s


class TestParseResults:
    def test_extracts_known_matches(self):
        out = HLTVScraper._parse_results(read("results.html"))
        assert len(out) == 2
        m = out[0]
        assert m.match_id == 2394174
        assert m.slug == "natus-vincere-vs-vitality-blast-premier-final"
        assert m.team1 == "Natus Vincere"
        assert m.team2 == "Vitality"
        assert m.event == "BLAST Premier Final"
        assert "2" in m.score and "1" in m.score
        assert m.stars == 3
        assert m.date_unix == 1750000000
        assert out[1].match_id == 2394175
        assert out[1].stars == 2

    def test_drifted_markup_parses_empty(self):
        assert HLTVScraper._parse_results(read("results_drifted.html")) == []

    def test_fetch_results_raises_on_drift_at_offset_zero(self):
        s = make_scraper(read("results_drifted.html"))
        with pytest.raises(ScrapeLayoutError):
            s.fetch_results(stars=3, offset=0)

    def test_fetch_results_empty_deep_offset_is_end_of_listing(self):
        s = make_scraper(read("results_drifted.html"))
        assert s.fetch_results(stars=3, offset=700) == []

    def test_fetch_results_ok_on_pristine_fixture(self):
        s = make_scraper(read("results.html"))
        assert len(s.fetch_results(stars=3, offset=0)) == 2


class TestParseMatch:
    def test_extracts_demo_url_and_maps(self):
        s = make_scraper(read("match.html"))
        md = s.fetch_match(2394174, "natus-vincere-vs-vitality")
        assert md.demo_url == "https://www.hltv.org/download/demo/98765"
        assert md.demo_id == 98765
        assert md.maps == ["mirage", "dust2", "inferno"]
        assert md.summary.team1 == "Natus Vincere"
        assert md.summary.team2 == "Vitality"

    def test_demo_not_released_is_none_not_error(self):
        s = make_scraper(read("match_no_demo.html"))
        md = s.fetch_match(2394174, "slug")
        assert md.demo_url is None
        assert md.demo_id is None

    def test_drifted_match_page_raises(self):
        s = make_scraper(read("match_drifted.html"))
        with pytest.raises(ScrapeLayoutError):
            s.fetch_match(2394174, "slug")
