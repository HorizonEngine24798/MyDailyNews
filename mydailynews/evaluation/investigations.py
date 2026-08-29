from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from mydailynews.evaluation.schema import EvalCorpus


INVESTIGATION_MODES = {
    "baseline",
    "retrieved_top3",
    "oracle_candidate",
    "oracle_ledger",
}


@dataclass(frozen=True)
class InvestigationCase:
    """Private diagnostic context for one source document.

    The packet deliberately omits canonical story IDs and expected decision
    labels. Oracle modes still use those private labels to select a correct
    prior document or expected fact, so their scores are capability ceilings,
    not production estimates.
    """

    prior_story_key: str = ""
    prior_document_ids: List[str] = field(default_factory=list)
    prior_title: str = ""
    prior_date: str = ""
    prior_fact_ids: List[str] = field(default_factory=list)
    prior_facts: List[str] = field(default_factory=list)
    current_fact_ids: List[str] = field(default_factory=list)
    current_facts: List[str] = field(default_factory=list)

    @property
    def has_prior_story(self) -> bool:
        return bool(self.prior_story_key and self.prior_document_ids)


@dataclass(frozen=True)
class EvaluationInvestigation:
    mode: str = "baseline"
    cases: Dict[tuple[str, str, str], InvestigationCase] = field(default_factory=dict)

    @property
    def uses_private_gold(self) -> bool:
        return self.mode in {"oracle_candidate", "oracle_ledger"}

    @property
    def production_comparable(self) -> bool:
        return not self.uses_private_gold

    def case_for(self, arc_id: str, date: str, document_id: str) -> InvestigationCase:
        return self.cases.get((arc_id, date, document_id), InvestigationCase())

    def disclosure(self) -> Dict[str, object]:
        supplied_fields: List[str] = []
        purpose = "Reproduce the existing broad-baseline model evaluation."
        if self.mode == "retrieved_top3":
            supplied_fields = ["lexically retrieved prior-story candidates"]
            purpose = "Measure the model after a gold-blind top-k candidate-retrieval stage."
        elif self.mode == "oracle_candidate":
            supplied_fields = ["correct prior-story candidate selected using private canonical identity"]
            purpose = "Measure identity and delta judgment when candidate retrieval is made perfect."
        elif self.mode == "oracle_ledger":
            supplied_fields = [
                "correct prior-story candidate selected using private canonical identity",
                "previously required fact texts",
                "current required fact texts when annotated",
            ]
            purpose = "Measure the model with an upper-bound source-fact ledger packet."
        return {
            "mode": self.mode,
            "uses_private_gold": self.uses_private_gold,
            "production_comparable": self.production_comparable,
            "purpose": purpose,
            "supplied_fields": supplied_fields,
            "case_packets": len(self.cases),
        }


def build_investigation(corpus: EvalCorpus, mode: str) -> EvaluationInvestigation:
    normalized = str(mode or "baseline").strip().lower()
    if normalized not in INVESTIGATION_MODES:
        raise ValueError(f"unsupported investigation mode: {mode!r}")
    if normalized in {"baseline", "retrieved_top3"}:
        return EvaluationInvestigation(mode=normalized)

    packets: Dict[tuple[str, str, str], InvestigationCase] = {}
    include_facts = normalized == "oracle_ledger"
    for arc in corpus.arcs:
        history: Dict[str, List[tuple[str, str, str]]] = {}
        shown_fact_ids: Dict[str, List[str]] = {}
        for day in arc.days:
            expectations = {item.document_id: item for item in day.expectations}
            for document in day.documents:
                expected = expectations[document.id]
                story_history = history.get(expected.canonical_story_id, [])
                prior_fact_ids = list(shown_fact_ids.get(expected.canonical_story_id, []))
                current_fact_ids = list(expected.required_fact_ids)
                if story_history:
                    first_document_id = story_history[0][0]
                    latest_document_id, latest_title, latest_date = story_history[-1]
                    packets[(arc.id, day.date, document.id)] = InvestigationCase(
                        prior_story_key=f"oracle:{first_document_id}",
                        prior_document_ids=[item[0] for item in story_history],
                        prior_title=latest_title,
                        prior_date=latest_date,
                        prior_fact_ids=prior_fact_ids if include_facts else [],
                        prior_facts=[arc.fact_catalog[item] for item in prior_fact_ids] if include_facts else [],
                        current_fact_ids=current_fact_ids if include_facts else [],
                        current_facts=[arc.fact_catalog[item] for item in current_fact_ids] if include_facts else [],
                    )
                else:
                    packets[(arc.id, day.date, document.id)] = InvestigationCase(
                        current_fact_ids=current_fact_ids if include_facts else [],
                        current_facts=[arc.fact_catalog[item] for item in current_fact_ids] if include_facts else [],
                    )
                history.setdefault(expected.canonical_story_id, []).append(
                    (document.id, document.title, day.date)
                )
                accumulated = shown_fact_ids.setdefault(expected.canonical_story_id, [])
                for fact_id in current_fact_ids:
                    if fact_id not in accumulated:
                        accumulated.append(fact_id)
    return EvaluationInvestigation(mode=normalized, cases=packets)
