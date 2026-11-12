from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from mydailynews.common.utils import is_consent_interstitial_text


DEFAULT_MULTILINGUAL_MODEL_ID = "intfloat/multilingual-e5-small"
REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_MULTILINGUAL_MODEL_DIR = REPO_ROOT / "models" / "multilingual-e5-small"
MAX_VECTOR_CACHE_SIZE = 4096

_MODEL_STATE: Dict[str, Any] = {
    "loaded": False,
    "available": False,
    "tokenizer": None,
    "model": None,
    "torch": None,
}
_VECTOR_CACHE: Dict[str, Any] = {}


def _load_model() -> bool:
    if _MODEL_STATE["loaded"]:
        return bool(_MODEL_STATE["available"])
    _MODEL_STATE["loaded"] = True
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        model_source: str | None = None
        local_only = True
        if (LOCAL_MULTILINGUAL_MODEL_DIR / "config.json").exists():
            model_source = str(LOCAL_MULTILINGUAL_MODEL_DIR)
        else:
            hf_hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--intfloat--multilingual-e5-small"
            if hf_hub.exists():
                model_source = DEFAULT_MULTILINGUAL_MODEL_ID
            elif os.environ.get("MYDAILYNEWS_ALLOW_MODEL_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
                model_source = DEFAULT_MULTILINGUAL_MODEL_ID
                local_only = False
        if not model_source:
            return False

        tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
        model = AutoModel.from_pretrained(model_source, local_files_only=local_only)
        model.eval()
        model.to("cpu")
        _MODEL_STATE["tokenizer"] = tokenizer
        _MODEL_STATE["model"] = model
        _MODEL_STATE["torch"] = torch
        _MODEL_STATE["available"] = True
        return True
    except Exception:
        _MODEL_STATE["available"] = False
        return False


def _cache_key(text: str, *, mode: str = "passage") -> str:
    cleaned = " ".join(str(text or "").split()).strip()[:520]
    digest = sha256(cleaned.encode("utf-8")).hexdigest()[:24]
    return digest if mode == "passage" else f"{mode}:{digest}"


def _normalize_passage(text: str, *, prefix: str = "passage") -> str:
    cleaned = " ".join(str(text or "").split()).strip()[:520]
    return f"{prefix}: {cleaned}"


def _prefetch_with_mode(texts: Iterable[str], *, mode: str = "passage") -> None:
    if not _load_model():
        return
    missing_keys: List[str] = []
    missing_passages: List[str] = []
    seen: set[str] = set()
    for raw in texts:
        cleaned = " ".join(str(raw or "").split()).strip()[:520]
        if not cleaned:
            continue
        key = _cache_key(cleaned, mode=mode)
        if key in _VECTOR_CACHE or key in seen:
            continue
        seen.add(key)
        missing_keys.append(key)
        missing_passages.append(_normalize_passage(cleaned, prefix=mode))

    if not missing_passages:
        return

    torch = _MODEL_STATE["torch"]
    tokenizer = _MODEL_STATE["tokenizer"]
    model = _MODEL_STATE["model"]

    if len(_VECTOR_CACHE) + len(missing_passages) > MAX_VECTOR_CACHE_SIZE:
        _VECTOR_CACHE.clear()

    batch_size = 32
    for start in range(0, len(missing_passages), batch_size):
        chunk_passages = missing_passages[start : start + batch_size]
        chunk_keys = missing_keys[start : start + batch_size]
        encoded = tokenizer(
            chunk_passages,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        with torch.inference_mode():
            outputs = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            pooled = torch.sum(outputs.last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        for key, vec in zip(chunk_keys, normalized):
            _VECTOR_CACHE[key] = vec


def prefetch_embeddings(texts: Iterable[str]) -> None:
    """Batch-encode and cache passage embeddings for candidate/story texts."""
    _prefetch_with_mode(texts, mode="passage")


def prefetch_query_embeddings(texts: Iterable[str]) -> None:
    """Batch-encode and cache query embeddings for preference/topic matching."""
    _prefetch_with_mode(texts, mode="query")


def _get_vector(text: str, *, mode: str = "passage") -> Any | None:
    cleaned = " ".join(str(text or "").split()).strip()[:520]
    if not cleaned or not _load_model():
        return None
    key = _cache_key(cleaned, mode=mode)
    vec = _VECTOR_CACHE.get(key)
    if vec is None:
        _prefetch_with_mode([cleaned], mode=mode)
        vec = _VECTOR_CACHE.get(key)
    return vec


def embedding_model_available() -> bool:
    """Return True when the local multilingual-e5-small model can run on CPU."""
    return bool(_load_model())


def _clean_context_text(candidate: Any, source_text: str) -> str:
    raw_source = "" if is_consent_interstitial_text(source_text) else str(source_text or "").strip()
    snippet = str(getattr(candidate, "snippet", "") or "").strip()
    clean_snippet = "" if is_consent_interstitial_text(snippet) else snippet
    return (raw_source or clean_snippet)[:420]


def candidate_record_texts(candidate: Any, source_text: str, records: Sequence[Any]) -> List[str]:
    texts: List[str] = []
    cand_title = str(getattr(candidate, "title", "") or "").strip()
    cand_context = _clean_context_text(candidate, source_text)
    if cand_title:
        texts.append(cand_title)
        texts.append(f"{cand_title}. {cand_context}" if cand_context else cand_title)

    for record in records:
        rec_title = str(getattr(record, "title", "") or "").strip()
        raw_aliases = list(getattr(record, "aliases", []) or [])
        aliases: List[str] = []
        seen_alias: set[str] = set()
        for value in [rec_title, *raw_aliases]:
            cleaned = str(value or "").strip()
            if cleaned and cleaned.casefold() not in seen_alias:
                seen_alias.add(cleaned.casefold())
                aliases.append(cleaned)
        aliases = aliases[-6:]
        texts.extend(aliases)

        facts = list(getattr(record, "facts", []) or [])[-8:]
        fact_snippets = [
            str(getattr(fact, "text", "") or "").strip()
            for fact in facts
            if str(getattr(fact, "kind", "") or "") != "headline"
            and str(getattr(fact, "text", "") or "").strip()
            and not is_consent_interstitial_text(str(getattr(fact, "text", "") or ""))
        ]
        fact_summary = " ".join(fact_snippets[:4])[:420]
        combined_header = ". ".join([rec_title, *aliases[:3]]).strip(". ")
        if combined_header:
            texts.append(f"{combined_header}. {fact_summary}" if fact_summary else combined_header)
        for alias in aliases[-3:]:
            texts.append(f"{alias}. {fact_summary}" if fact_summary else alias)
    return texts


def semantic_story_similarity(
    candidate: Any,
    *,
    source_text: str = "",
    record: Any,
) -> tuple[float, float, float]:
    """Return (title_cosine, full_cosine, max_cosine) using CPU multilingual-e5-small."""
    if not _load_model():
        return 0.0, 0.0, 0.0

    torch = _MODEL_STATE["torch"]
    cand_title = str(getattr(candidate, "title", "") or "").strip()
    cand_context = _clean_context_text(candidate, source_text)
    cand_full = f"{cand_title}. {cand_context}" if cand_context else cand_title

    v_cand_title = _get_vector(cand_title)
    v_cand_full = _get_vector(cand_full)
    if v_cand_title is None or v_cand_full is None:
        return 0.0, 0.0, 0.0

    rec_title = str(getattr(record, "title", "") or "").strip()
    raw_aliases = list(getattr(record, "aliases", []) or [])
    aliases: List[str] = []
    seen_alias: set[str] = set()
    for value in [rec_title, *raw_aliases]:
        cleaned = str(value or "").strip()
        if cleaned and cleaned.casefold() not in seen_alias:
            seen_alias.add(cleaned.casefold())
            aliases.append(cleaned)
    aliases = aliases[-6:]

    title_sims: List[float] = []
    for alias in aliases:
        v_alias = _get_vector(alias)
        if v_alias is not None:
            title_sims.append(float(torch.dot(v_cand_title, v_alias).item()))

    facts = list(getattr(record, "facts", []) or [])[-8:]
    fact_snippets = [
        str(getattr(fact, "text", "") or "").strip()
        for fact in facts
        if str(getattr(fact, "kind", "") or "") != "headline"
        and str(getattr(fact, "text", "") or "").strip()
        and not is_consent_interstitial_text(str(getattr(fact, "text", "") or ""))
    ]
    fact_summary = " ".join(fact_snippets[:4])[:420]
    full_sims: List[float] = []
    combined_header = ". ".join([rec_title, *aliases[:3]]).strip(". ")
    if combined_header:
        v_combined = _get_vector(f"{combined_header}. {fact_summary}" if fact_summary else combined_header)
        if v_combined is not None:
            full_sims.append(float(torch.dot(v_cand_full, v_combined).item()))
    for alias in aliases[-3:]:
        rec_full = f"{alias}. {fact_summary}" if fact_summary else alias
        v_full = _get_vector(rec_full)
        if v_full is not None:
            full_sims.append(float(torch.dot(v_cand_full, v_full).item()))

    best_title = max(title_sims) if title_sims else 0.0
    best_full = max(full_sims) if full_sims else 0.0
    return best_title, best_full, max(best_title, best_full)


def _dot_pair(text_a: str, text_b: str, *, mode_a: str = "passage", mode_b: str = "passage") -> float:
    cleaned_a = " ".join(str(text_a or "").split()).strip()
    cleaned_b = " ".join(str(text_b or "").split()).strip()
    if not cleaned_a or not cleaned_b or not _load_model():
        return 0.0
    v_a = _get_vector(cleaned_a, mode=mode_a)
    v_b = _get_vector(cleaned_b, mode=mode_b)
    torch = _MODEL_STATE["torch"]
    if v_a is None or v_b is None or torch is None:
        return 0.0
    return float(torch.dot(v_a, v_b).item())


def calculate_text_similarity(text_a: str, text_b: str) -> float:
    """Return symmetric passage-to-passage cosine similarity."""
    return _dot_pair(text_a, text_b, mode_a="passage", mode_b="passage")


def query_passage_similarity(query_text: str, passage_text: str) -> float:
    """Return asymmetric query-to-passage E5 cosine similarity."""
    return _dot_pair(query_text, passage_text, mode_a="query", mode_b="passage")


def batch_query_passage_similarities(
    queries: Sequence[str],
    passages: Sequence[str],
) -> Dict[tuple[str, str], float]:
    """Compute query-by-passage E5 cosine matrix in one batched inference pass."""
    clean_queries = list(dict.fromkeys(" ".join(str(item or "").split()).strip() for item in queries if str(item or "").strip()))
    clean_passages = list(dict.fromkeys(" ".join(str(item or "").split()).strip() for item in passages if str(item or "").strip()))
    if not clean_queries or not clean_passages or not _load_model():
        return {}

    prefetch_query_embeddings(clean_queries)
    prefetch_embeddings(clean_passages)

    torch = _MODEL_STATE["torch"]
    if torch is None:
        return {}

    q_items: List[tuple[str, Any]] = []
    for query in clean_queries:
        vec = _get_vector(query, mode="query")
        if vec is not None:
            q_items.append((query, vec))

    p_items: List[tuple[str, Any]] = []
    for passage in clean_passages:
        vec = _get_vector(passage, mode="passage")
        if vec is not None:
            p_items.append((passage, vec))

    if not q_items or not p_items:
        return {}

    with torch.inference_mode():
        q_matrix = torch.stack([vec for _, vec in q_items], dim=0)
        p_matrix = torch.stack([vec for _, vec in p_items], dim=0)
        sims = torch.matmul(q_matrix, p_matrix.transpose(0, 1)).tolist()

    output: Dict[tuple[str, str], float] = {}
    for row_idx, (query, _) in enumerate(q_items):
        row = sims[row_idx]
        for col_idx, (passage, _) in enumerate(p_items):
            output[(query, passage)] = float(row[col_idx])
    return output


