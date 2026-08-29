from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List

from mydailynews.evaluation.schema import (
    DELTA_TYPES,
    DISPLAYS,
    RELATIONSHIPS,
    RELEVANCE_LABELS,
    EvalCorpus,
    EvalExpectation,
    EvalPrediction,
)


EVAL_REPORT_SCHEMA_VERSION = "change_monitor.eval_report.v2"


def score_predictions(corpus: EvalCorpus, predictions: Iterable[EvalPrediction]) -> Dict[str, Any]:
    rows = list(predictions)
    expected_by_key = corpus.expectations_by_key()
    document_key_by_arc_and_id = {
        (key[0], key[2]): key
        for key in expected_by_key
    }
    document_order = {
        (arc.id, document.id): (day_index, document_index)
        for arc in corpus.arcs
        for day_index, day in enumerate(arc.days)
        for document_index, document in enumerate(day.documents)
    }
    prediction_by_key: Dict[tuple[str, str, str], EvalPrediction] = {}
    duplicate_keys: List[str] = []
    for prediction in rows:
        key = (prediction.arc_id, prediction.date, prediction.document_id)
        if key in prediction_by_key:
            duplicate_keys.append("/".join(key))
            continue
        prediction_by_key[key] = prediction

    expected_keys = set(expected_by_key)
    predicted_keys = set(prediction_by_key)
    missing_keys = sorted(expected_keys.difference(predicted_keys))
    extra_keys = sorted(predicted_keys.difference(expected_keys))
    effective_predictions = dict(prediction_by_key)
    for key in missing_keys:
        effective_predictions[key] = _missing_prediction(key)

    arc_by_id = {arc.id: arc for arc in corpus.arcs}
    known_fact_ids = {arc.id: set(arc.fact_catalog) for arc in corpus.arcs}
    overall = _score_subset(expected_by_key, effective_predictions, expected_keys, known_fact_ids)
    quiet_days = corpus.quiet_days()
    overall.update(_quiet_day_metrics(quiet_days, prediction_by_key))
    by_tag: Dict[str, Dict[str, Any]] = {}
    tag_keys: Dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    split_keys: Dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    tag_quiet_days: Dict[str, set[tuple[str, str]]] = defaultdict(set)
    split_quiet_days: Dict[str, set[tuple[str, str]]] = defaultdict(set)
    for key in expected_keys:
        arc = arc_by_id[key[0]]
        split_keys[arc.split].add(key)
        for tag in arc.tags:
            tag_keys[tag].add(key)
    for arc in corpus.arcs:
        for day in arc.days:
            if day.documents:
                continue
            pair = (arc.id, day.date)
            split_quiet_days[arc.split].add(pair)
            for tag in arc.tags:
                tag_quiet_days[tag].add(pair)
    for tag, keys in sorted(tag_keys.items()):
        by_tag[tag] = _score_subset(expected_by_key, effective_predictions, keys, known_fact_ids)
        by_tag[tag].update(_quiet_day_metrics(tag_quiet_days.get(tag, set()), prediction_by_key))
    by_split = {
        split: _score_subset(expected_by_key, effective_predictions, keys, known_fact_ids)
        for split, keys in sorted(split_keys.items())
    }
    for split, metrics in by_split.items():
        metrics.update(_quiet_day_metrics(split_quiet_days.get(split, set()), prediction_by_key))

    validation_errors = []
    if duplicate_keys:
        validation_errors.append(f"duplicate prediction keys: {sorted(duplicate_keys)}")
    if extra_keys:
        validation_errors.append(f"unknown prediction keys: {[('/'.join(key)) for key in extra_keys]}")
    for key in sorted(expected_keys.intersection(predicted_keys)):
        prediction = prediction_by_key[key]
        label = "/".join(key)
        private_candidate_intervention = str(
            prediction.metadata.get("investigation_mode", "") or ""
        ) in {"oracle_candidate", "oracle_ledger"}
        if not prediction.predicted_story_id:
            validation_errors.append(f"{label}: predicted_story_id is empty")
        if prediction.relationship not in RELATIONSHIPS:
            validation_errors.append(f"{label}: invalid relationship {prediction.relationship!r}")
        if prediction.delta_type not in DELTA_TYPES:
            validation_errors.append(f"{label}: invalid delta_type {prediction.delta_type!r}")
        if prediction.display not in DISPLAYS:
            validation_errors.append(f"{label}: invalid display {prediction.display!r}")
        if prediction.profile_relevance not in RELEVANCE_LABELS:
            validation_errors.append(f"{label}: invalid profile_relevance {prediction.profile_relevance!r}")
        if not isinstance(prediction.material, bool):
            validation_errors.append(f"{label}: material must be boolean")
        if not isinstance(prediction.selected, bool):
            validation_errors.append(f"{label}: selected must be boolean")
        if prediction.reported_fact_ids is not None:
            unknown_facts = set(prediction.reported_fact_ids).difference(known_fact_ids.get(key[0], set()))
            if unknown_facts:
                validation_errors.append(f"{label}: unknown reported fact IDs {sorted(unknown_facts)}")
        if "candidate_prior_stories" in prediction.metadata:
            candidates = prediction.metadata.get("candidate_prior_stories")
            if not isinstance(candidates, list):
                validation_errors.append(f"{label}: candidate_prior_stories must be an array")
            else:
                for candidate_index, candidate in enumerate(candidates):
                    if not isinstance(candidate, dict) or not isinstance(candidate.get("document_ids"), list):
                        validation_errors.append(
                            f"{label}: candidate_prior_stories[{candidate_index}] needs a document_ids array"
                        )
                        continue
                    story_key = str(candidate.get("story_key", "") or "").strip()
                    if not story_key:
                        validation_errors.append(
                            f"{label}: candidate_prior_stories[{candidate_index}] needs a story_key"
                        )
                    for document_id in candidate["document_ids"]:
                        prior_key = document_key_by_arc_and_id.get((key[0], str(document_id or "")))
                        if prior_key is None:
                            validation_errors.append(
                                f"{label}: candidate_prior_stories[{candidate_index}] references unknown document "
                                f"{document_id!r}"
                            )
                        elif document_order[(key[0], str(document_id))] >= document_order[(key[0], key[2])]:
                            validation_errors.append(
                                f"{label}: candidate prior document {document_id!r} is not earlier in corpus chronology"
                            )
                        elif prior_key not in prediction_by_key:
                            validation_errors.append(
                                f"{label}: candidate prior document {document_id!r} has no prediction to verify"
                            )
                        elif (
                            not private_candidate_intervention
                            and prediction_by_key[prior_key].predicted_story_id != story_key
                        ):
                            validation_errors.append(
                                f"{label}: candidate story_key {story_key!r} does not match prior prediction "
                                f"for document {document_id!r}"
                            )
    warnings = []
    if overall["claim_evaluation_coverage"] < 1.0:
        warnings.append(
            "Claim-level faithfulness is only scored for predictions with reported_fact_ids; "
            "decision metrics remain complete."
        )
    return {
        "schema_version": EVAL_REPORT_SCHEMA_VERSION,
        "corpus": {
            "schema_version": corpus.schema_version,
            "name": corpus.name,
            "arcs": len(corpus.arcs),
            "documents": len(expected_keys),
            "quiet_days": len(quiet_days),
        },
        "prediction_counts": {
            "received": len(rows),
            "unique": len(prediction_by_key),
            "missing": len(missing_keys),
            "extra": len(extra_keys),
            "duplicates": len(duplicate_keys),
            "quiet_day_outputs": sum(
                1
                for key in prediction_by_key
                if (key[0], key[1]) in quiet_days
            ),
            "model_fallback_used": sum(
                1 for prediction in rows if prediction.metadata.get("model_fallback_used") is True
            ),
            "model_error_predictions": sum(
                1 for prediction in rows if bool(prediction.metadata.get("model_error"))
            ),
        },
        "overall": overall,
        "by_tag": by_tag,
        "by_split": by_split,
        "validation": {
            "errors": validation_errors,
            "warnings": warnings,
            "missing_keys": ["/".join(key) for key in missing_keys],
        },
    }


def _score_subset(
    expected_by_key: Dict[tuple[str, str, str], EvalExpectation],
    predictions: Dict[tuple[str, str, str], EvalPrediction],
    keys: Iterable[tuple[str, str, str]],
    known_fact_ids: Dict[str, set[str]],
) -> Dict[str, Any]:
    ordered_keys = sorted(keys)
    if not ordered_keys:
        return {"cases": 0}
    expected = [expected_by_key[key] for key in ordered_keys]
    actual = [predictions[key] for key in ordered_keys]

    relationship_accuracy = _accuracy(
        [item.relationship for item in expected],
        [item.relationship for item in actual],
    )
    delta_accuracy = _accuracy(
        [item.delta_type for item in expected],
        [item.delta_type for item in actual],
    )
    display_accuracy = _accuracy(
        [item.display for item in expected],
        [item.display for item in actual],
    )
    relevance_accuracy = _accuracy(
        [item.profile_relevance for item in expected],
        [item.profile_relevance for item in actual],
    )
    selection = _binary_metrics(
        [item.should_select for item in expected],
        [item.selected for item in actual],
    )
    material = _binary_metrics(
        [item.material for item in expected],
        [item.material for item in actual],
    )
    novelty_indexes = [
        index
        for index, item in enumerate(expected)
        if item.delta_type != "uncertain"
    ]
    novelty = _binary_metrics(
        [expected[index].delta_type != "unchanged" for index in novelty_indexes],
        [actual[index].delta_type not in {"unchanged", "uncertain"} for index in novelty_indexes],
    )
    identity = _identity_metrics(ordered_keys, expected_by_key, predictions)

    material_selectable = [
        index
        for index, item in enumerate(expected)
        if item.material and item.should_select
    ]
    false_suppression = sum(
        1
        for index in material_selectable
        if actual[index].display == "omit" or not actual[index].selected
    )
    expected_omissions = [index for index, item in enumerate(expected) if item.display == "omit"]
    repeated_full = sum(
        1
        for index in expected_omissions
        if actual[index].display == "full_report" and actual[index].selected
    )
    must_select = [index for index, item in enumerate(expected) if item.profile_relevance == "must_select"]
    irrelevant = [index for index, item in enumerate(expected) if item.profile_relevance == "irrelevant"]
    continuation_indexes = [
        index
        for index, item in enumerate(expected)
        if item.relationship == "same_story"
    ]
    material_continuations = [index for index in continuation_indexes if expected[index].material]

    claim_indexes = [index for index, item in enumerate(actual) if item.reported_fact_ids is not None]
    required_total = 0
    required_found = 0
    forbidden_case_violations = 0
    unsupported_case_violations = 0
    faithfulness_passes = 0
    for index in claim_indexes:
        reported = set(actual[index].reported_fact_ids or [])
        required = set(expected[index].required_fact_ids)
        forbidden = set(expected[index].forbidden_fact_ids)
        required_total += len(required)
        required_found += len(required.intersection(reported))
        forbidden_hit = bool(forbidden.intersection(reported))
        unknown_reported = reported.difference(known_fact_ids.get(ordered_keys[index][0], set()))
        unsupported_hit = bool(actual[index].unsupported_claims or unknown_reported)
        forbidden_case_violations += int(forbidden_hit)
        unsupported_case_violations += int(unsupported_hit)
        faithfulness_passes += int(not forbidden_hit and not unsupported_hit)

    latencies = sorted(max(0.0, float(item.latency_ms)) for item in actual)
    instruction_following = round((display_accuracy + selection["accuracy"]) / 2.0, 4)
    stage_diagnostics = _stage_diagnostics(
        ordered_keys,
        expected_by_key,
        expected,
        actual,
    )
    return {
        "cases": len(ordered_keys),
        "story_identity_pairwise_precision": identity["precision"],
        "story_identity_pairwise_recall": identity["recall"],
        "story_identity_pairwise_f1": identity["f1"],
        "relationship_accuracy": relationship_accuracy,
        "delta_type_accuracy": delta_accuracy,
        "continuation_delta_type_accuracy": _accuracy(
            [expected[index].delta_type for index in continuation_indexes],
            [actual[index].delta_type for index in continuation_indexes],
        ) if continuation_indexes else 1.0,
        "material_update_precision": material["precision"],
        "material_update_recall": material["recall"],
        "material_update_f1": material["f1"],
        "novelty_detection_accuracy": novelty["accuracy"],
        "novelty_detection_precision": novelty["precision"],
        "novelty_detection_recall": novelty["recall"],
        "novelty_detection_f1": novelty["f1"],
        "material_continuation_recall": _rate(
            sum(1 for index in material_continuations if actual[index].material),
            len(material_continuations),
        ),
        "linked_material_continuation_recall": _rate(
            sum(
                1
                for index in material_continuations
                if actual[index].material and actual[index].relationship == "same_story"
            ),
            len(material_continuations),
        ),
        "display_policy_accuracy": display_accuracy,
        "selection_precision": selection["precision"],
        "selection_recall": selection["recall"],
        "selection_f1": selection["f1"],
        "profile_relevance_accuracy": relevance_accuracy,
        "must_select_recall": _rate(sum(1 for index in must_select if actual[index].selected), len(must_select)),
        "irrelevant_selection_rate": _rate(sum(1 for index in irrelevant if actual[index].selected), len(irrelevant)),
        "false_suppression_rate": _rate(false_suppression, len(material_selectable)),
        "unchanged_full_report_rate": _rate(repeated_full, len(expected_omissions)),
        "instruction_following_score": instruction_following,
        "claim_evaluation_coverage": _rate(len(claim_indexes), len(actual)),
        "required_fact_recall": (
            _rate(required_found, required_total) if claim_indexes and required_total else (1.0 if claim_indexes else None)
        ),
        "forbidden_fact_case_rate": _rate(forbidden_case_violations, len(claim_indexes)) if claim_indexes else None,
        "unsupported_claim_case_rate": _rate(unsupported_case_violations, len(claim_indexes)) if claim_indexes else None,
        "faithfulness_pass_rate": _rate(faithfulness_passes, len(claim_indexes)) if claim_indexes else None,
        "latency_ms_mean": round(sum(latencies) / max(1, len(latencies)), 4),
        "latency_ms_p50": _percentile(latencies, 0.5),
        "latency_ms_p95": _percentile(latencies, 0.95),
        "candidate_metadata_coverage": stage_diagnostics["candidate_retrieval"]["metadata_coverage"],
        "candidate_recall_at_1": stage_diagnostics["candidate_retrieval"]["recall_at_1"],
        "candidate_recall_at_3": stage_diagnostics["candidate_retrieval"]["recall_at_3"],
        "candidate_recall_at_limit": stage_diagnostics["candidate_retrieval"]["recall_at_limit"],
        "relationship_accuracy_given_candidate": stage_diagnostics["identity"]["relationship_accuracy_given_correct_candidate"],
        "same_story_link_accuracy_given_candidate": stage_diagnostics["identity"]["same_story_link_accuracy_given_correct_candidate"],
        "continuation_delta_accuracy_given_correct_identity": stage_diagnostics["delta"]["continuation_accuracy_given_correct_identity"],
        "display_accuracy_given_correct_semantics": stage_diagnostics["policy"]["display_accuracy_given_correct_semantics"],
        "stage_diagnostics": stage_diagnostics,
    }


def _stage_diagnostics(
    ordered_keys: List[tuple[str, str, str]],
    expected_by_key: Dict[tuple[str, str, str], EvalExpectation],
    expected: List[EvalExpectation],
    actual: List[EvalPrediction],
) -> Dict[str, Any]:
    expectation_by_document = {
        (key[0], key[2]): item
        for key, item in expected_by_key.items()
    }
    candidate_rows = [_candidate_rows(item) for item in actual]
    metadata_indexes = [index for index, rows in enumerate(candidate_rows) if rows is not None]
    continuation_indexes = [
        index for index, item in enumerate(expected) if item.relationship == "same_story"
    ]
    new_story_indexes = [
        index for index, item in enumerate(expected) if item.relationship == "new_story"
    ]
    continuation_with_metadata = [
        index for index in continuation_indexes if candidate_rows[index] is not None
    ]
    has_candidate_metadata = bool(metadata_indexes)

    def candidate_rank(index: int) -> int | None:
        rows = candidate_rows[index]
        if rows is None:
            return None
        target_story = expected[index].canonical_story_id
        arc_id = ordered_keys[index][0]
        for rank, row in enumerate(rows, start=1):
            for document_id in row.get("document_ids", []):
                prior = expectation_by_document.get((arc_id, document_id))
                if prior is not None and prior.canonical_story_id == target_story:
                    return rank
        return None

    def correct_candidate_story_keys(index: int) -> set[str]:
        rows = candidate_rows[index]
        if rows is None:
            return set()
        target_story = expected[index].canonical_story_id
        arc_id = ordered_keys[index][0]
        return {
            str(row.get("story_key", "") or "")
            for row in rows
            if str(row.get("story_key", "") or "")
            and any(
                (
                    expectation_by_document.get((arc_id, document_id)) is not None
                    and expectation_by_document[(arc_id, document_id)].canonical_story_id == target_story
                )
                for document_id in row.get("document_ids", [])
            )
        }

    correct_candidate_ranks = {
        index: candidate_rank(index)
        for index in continuation_indexes
    }
    correct_candidate_keys = {
        index: correct_candidate_story_keys(index)
        for index in continuation_indexes
    }
    candidate_counts = [len(candidate_rows[index] or []) for index in metadata_indexes]
    reciprocal_ranks = [
        1.0 / rank
        for rank in correct_candidate_ranks.values()
        if isinstance(rank, int) and rank > 0
    ]
    candidate_hit_indexes = [
        index for index, rank in correct_candidate_ranks.items() if rank is not None
    ]
    correct_relationship_indexes = [
        index
        for index, item in enumerate(expected)
        if actual[index].relationship == item.relationship
    ]
    correctly_linked_continuation_indexes = [
        index
        for index in candidate_hit_indexes
        if actual[index].relationship == "same_story"
        and actual[index].predicted_story_id in correct_candidate_keys[index]
    ]
    supplied_story_keys = [
        {
            str(row.get("story_key", "") or "")
            for row in (candidate_rows[index] or [])
            if str(row.get("story_key", "") or "")
        }
        for index in range(len(actual))
    ]

    def identity_is_correct(index: int) -> bool:
        if expected[index].relationship == "same_story":
            return index in correctly_linked_continuation_indexes
        return (
            actual[index].relationship == expected[index].relationship
            and actual[index].predicted_story_id not in supplied_story_keys[index]
        )

    correct_semantic_indexes = [
        index
        for index, item in enumerate(expected)
        if identity_is_correct(index)
        and actual[index].delta_type == item.delta_type
        and actual[index].material == item.material
        and actual[index].profile_relevance == item.profile_relevance
    ]
    oracle_ledger_indexes = [
        index
        for index, item in enumerate(actual)
        if item.metadata.get("investigation_mode") == "oracle_ledger"
    ]
    oracle_ledger_continuations = [
        index for index in oracle_ledger_indexes if expected[index].relationship == "same_story"
    ]
    oracle_ledger_current_facts = [
        index
        for index in oracle_ledger_indexes
        if int(actual[index].metadata.get("oracle_current_fact_count", 0) or 0) > 0
    ]

    return {
        "candidate_retrieval": {
            "metadata_cases": len(metadata_indexes),
            "metadata_missing_cases": len(actual) - len(metadata_indexes),
            "metadata_coverage": _optional_rate(len(metadata_indexes), len(actual)),
            "continuation_cases": len(continuation_indexes),
            "continuation_cases_with_metadata": len(continuation_with_metadata),
            "correct_candidate_cases": len(candidate_hit_indexes),
            "recall_at_1": _optional_rate(
                sum(1 for rank in correct_candidate_ranks.values() if rank == 1),
                len(continuation_indexes) if has_candidate_metadata else 0,
            ),
            "recall_at_3": _optional_rate(
                sum(1 for rank in correct_candidate_ranks.values() if rank is not None and rank <= 3),
                len(continuation_indexes) if has_candidate_metadata else 0,
            ),
            "recall_at_limit": _optional_rate(
                len(candidate_hit_indexes),
                len(continuation_indexes) if has_candidate_metadata else 0,
            ),
            "mean_reciprocal_rank": (
                round(sum(reciprocal_ranks) / len(continuation_indexes), 4)
                if has_candidate_metadata and continuation_indexes
                else None
            ),
            "mean_candidate_count": (
                round(sum(candidate_counts) / len(candidate_counts), 4)
                if candidate_counts
                else None
            ),
            "p95_candidate_count": _percentile(sorted(candidate_counts), 0.95) if candidate_counts else None,
            "new_story_no_candidate_rate": _optional_rate(
                sum(
                    1
                    for index in new_story_indexes
                    if candidate_rows[index] is not None and not candidate_rows[index]
                ),
                len(new_story_indexes) if has_candidate_metadata else 0,
            ),
        },
        "identity": {
            "cases_with_correct_candidate": len(candidate_hit_indexes),
            "relationship_accuracy_given_correct_candidate": _optional_rate(
                sum(
                    1
                    for index in candidate_hit_indexes
                    if actual[index].relationship == expected[index].relationship
                ),
                len(candidate_hit_indexes),
            ),
            "same_story_recall_given_correct_candidate": _optional_rate(
                sum(1 for index in candidate_hit_indexes if actual[index].relationship == "same_story"),
                len(candidate_hit_indexes),
            ),
            "same_story_link_accuracy_given_correct_candidate": _optional_rate(
                len(correctly_linked_continuation_indexes),
                len(candidate_hit_indexes),
            ),
            "new_story_overmerge_rate": _optional_rate(
                sum(1 for index in new_story_indexes if actual[index].relationship == "same_story"),
                len(new_story_indexes),
            ),
        },
        "delta": {
            "cases_with_correct_relationship": len(correct_relationship_indexes),
            "accuracy_given_correct_relationship": _optional_rate(
                sum(
                    1
                    for index in correct_relationship_indexes
                    if actual[index].delta_type == expected[index].delta_type
                ),
                len(correct_relationship_indexes),
            ),
            "continuation_cases_with_correct_identity": len(correctly_linked_continuation_indexes),
            "continuation_accuracy_given_correct_identity": _optional_rate(
                sum(
                    1
                    for index in correctly_linked_continuation_indexes
                    if actual[index].delta_type == expected[index].delta_type
                ),
                len(correctly_linked_continuation_indexes),
            ),
        },
        "policy": {
            "cases_with_correct_semantics": len(correct_semantic_indexes),
            "display_accuracy_given_correct_semantics": _optional_rate(
                sum(
                    1
                    for index in correct_semantic_indexes
                    if actual[index].display == expected[index].display
                ),
                len(correct_semantic_indexes),
            ),
            "selection_accuracy_given_correct_semantics": _optional_rate(
                sum(
                    1
                    for index in correct_semantic_indexes
                    if actual[index].selected == expected[index].should_select
                ),
                len(correct_semantic_indexes),
            ),
        },
        "oracle_fact_packet": {
            "cases": len(oracle_ledger_indexes),
            "continuation_prior_fact_coverage": _optional_rate(
                sum(
                    1
                    for index in oracle_ledger_continuations
                    if int(actual[index].metadata.get("oracle_prior_fact_count", 0) or 0) > 0
                ),
                len(oracle_ledger_continuations),
            ),
            "current_fact_coverage": _optional_rate(
                len(oracle_ledger_current_facts),
                len(oracle_ledger_indexes),
            ),
        },
    }


def _candidate_rows(prediction: EvalPrediction) -> List[Dict[str, Any]] | None:
    if "candidate_prior_stories" not in prediction.metadata:
        return None
    raw = prediction.metadata.get("candidate_prior_stories")
    if not isinstance(raw, list):
        return []
    output: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        document_ids = item.get("document_ids", [])
        if not isinstance(document_ids, list):
            document_ids = []
        output.append(
            {
                "story_key": str(item.get("story_key", "") or ""),
                "document_ids": [str(value) for value in document_ids if str(value).strip()],
            }
        )
    return output


def _identity_metrics(
    keys: List[tuple[str, str, str]],
    expected_by_key: Dict[tuple[str, str, str], EvalExpectation],
    predictions: Dict[tuple[str, str, str], EvalPrediction],
) -> Dict[str, float]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    for left_index, left_key in enumerate(keys):
        for right_key in keys[left_index + 1 :]:
            if left_key[0] != right_key[0]:
                continue
            gold_same = (
                expected_by_key[left_key].canonical_story_id
                == expected_by_key[right_key].canonical_story_id
            )
            predicted_same = (
                predictions[left_key].predicted_story_id
                == predictions[right_key].predicted_story_id
            )
            if gold_same and predicted_same:
                true_positive += 1
            elif not gold_same and predicted_same:
                false_positive += 1
            elif gold_same and not predicted_same:
                false_negative += 1
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


def _binary_metrics(expected: List[bool], actual: List[bool]) -> Dict[str, float]:
    true_positive = sum(1 for gold, predicted in zip(expected, actual) if gold and predicted)
    false_positive = sum(1 for gold, predicted in zip(expected, actual) if not gold and predicted)
    false_negative = sum(1 for gold, predicted in zip(expected, actual) if gold and not predicted)
    correct = sum(1 for gold, predicted in zip(expected, actual) if gold == predicted)
    precision = _rate(true_positive, true_positive + false_positive)
    recall = _rate(true_positive, true_positive + false_negative)
    return {
        "accuracy": _rate(correct, len(expected)),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
    }


def _accuracy(expected: List[Any], actual: List[Any]) -> float:
    return _rate(sum(1 for gold, predicted in zip(expected, actual) if gold == predicted), len(expected))


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(float(numerator) / float(denominator), 4)


def _optional_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return round(2.0 * precision * recall / (precision + recall), 4)


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    index = max(0, min(len(values) - 1, int(round((len(values) - 1) * fraction))))
    return round(values[index], 4)


def _missing_prediction(key: tuple[str, str, str]) -> EvalPrediction:
    return EvalPrediction(
        arc_id=key[0],
        date=key[1],
        document_id=key[2],
        predicted_story_id=f"__missing__:{'/'.join(key)}",
        relationship="uncertain",
        delta_type="uncertain",
        material=False,
        display="full_report",
        profile_relevance="eligible",
        selected=False,
        reported_fact_ids=None,
        metadata={"missing_prediction": True},
    )


def _quiet_day_metrics(
    quiet_days: set[tuple[str, str]],
    predictions: Dict[tuple[str, str, str], EvalPrediction],
) -> Dict[str, Any]:
    output_days = {
        (key[0], key[1])
        for key in predictions
        if (key[0], key[1]) in quiet_days
    }
    # With no quiet-day opportunities there can be no false quiet-day output.
    # The generic _rate helper returns 1.0 for an empty denominator because that
    # convention is useful for recall, but it is the wrong neutral value here.
    false_output_rate = _rate(len(output_days), len(quiet_days)) if quiet_days else 0.0
    return {
        "quiet_days": len(quiet_days),
        "quiet_day_outputs": sum(
            1
            for key in predictions
            if (key[0], key[1]) in quiet_days
        ),
        "quiet_day_false_output_rate": false_output_rate,
        "quiet_day_abstention_rate": round(1.0 - false_output_rate, 4),
    }
