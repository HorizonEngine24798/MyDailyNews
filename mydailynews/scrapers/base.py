from __future__ import annotations

from datetime import datetime
from typing import List, Protocol

from ..models import NewsCandidate


class HeadlineScraper(Protocol):
    def fetch(self, since: datetime) -> List[NewsCandidate]:
        """Return headline-level candidates newer than the provided timestamp."""
