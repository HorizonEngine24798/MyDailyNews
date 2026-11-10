import json
import sys
import unittest
from collections import Counter
from pathlib import Path

from mydailynews.perspectives.sources import load_source_registry

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from audit_rss_sources import dump_registry  # noqa: E402

G20_MEMBERS = ["AR", "AU", "BR", "CA", "CN", "FR", "DE", "IN", "ID", "IT", "JP", "MX", "RU", "SA", "ZA", "KR", "TR", "GB", "US", "EU"]


class SourceRegistryCoverageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = load_source_registry()
        self.enabled = [source for source in self.sources if source.get("enabled", True)]

    def test_every_g20_member_has_at_least_three_enabled_sources(self) -> None:
        counts = Counter(str(source["country"]).upper() for source in self.enabled)
        thin = {code: counts.get(code, 0) for code in G20_MEMBERS if counts.get(code, 0) < 3}
        self.assertEqual(thin, {}, f"G20 members below 3 enabled sources: {thin}")

    def test_g20_members_have_native_or_multiple_outlet_coverage(self) -> None:
        for code in G20_MEMBERS:
            outlets = {str(source["homepage_url"]).split("/")[2].removeprefix("www.") for source in self.enabled if source["country"] == code}
            self.assertGreaterEqual(len(outlets), 3, f"{code} needs at least 3 distinct outlets, got {sorted(outlets)}")

    def test_bbc_language_services_are_attributed_to_gb(self) -> None:
        for source in self.sources:
            if "bbc" in str(source["homepage_url"]).lower() or "bbci.co.uk" in " ".join(source.get("feed_urls") or []):
                self.assertEqual(source["country"], "GB", source["source_id"])

    def test_registry_file_matches_canonical_serialization(self) -> None:
        path = ROOT / "mydailynews" / "data" / "sources.rss.json"
        raw = path.read_bytes().decode("utf-8")
        self.assertEqual(raw, dump_registry(json.loads(raw)))


if __name__ == "__main__":
    unittest.main()
