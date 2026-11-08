from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from mydailynews.common.utils import compact_json, normalize_whitespace
from mydailynews.perspectives.schemas import MULTILINGUAL_QUERY_ALIGNMENT_SCHEMA


# Supported language code aliases
LANGUAGE_CODE_MAP = {
    "en": "en",
    "english": "en",
    "fr": "fr",
    "french": "fr",
    "de": "de",
    "german": "de",
    "es": "es",
    "spanish": "es",
    "it": "it",
    "italian": "it",
    "pt": "pt",
    "portuguese": "pt",
    "ja": "ja",
    "japanese": "ja",
    "zh": "zh",
    "chinese": "zh",
    "ru": "ru",
    "russian": "ru",
    "ar": "ar",
    "arabic": "ar",
    "ko": "ko",
    "korean": "ko",
}

# Country code to primary national language mapping
COUNTRY_DEFAULT_LANGUAGES = {
    "US": "en",
    "GB": "en",
    "UK": "en",
    "CA": "en",
    "AU": "en",
    "NZ": "en",
    "IE": "en",
    "FR": "fr",
    "BE": "fr",
    "DE": "de",
    "AT": "de",
    "CH": "de",
    "ES": "es",
    "MX": "es",
    "AR": "es",
    "CL": "es",
    "CO": "es",
    "IT": "it",
    "PT": "pt",
    "BR": "pt",
    "JP": "ja",
    "CN": "zh",
    "TW": "zh",
    "HK": "zh",
    "RU": "ru",
    "KR": "ko",
    "SA": "ar",
    "AE": "ar",
    "EG": "ar",
    "IN": "en",
}

# Assist lookup reference for standard international institutions and acronyms
MULTILINGUAL_ASSIST_LOOKUP: Dict[str, Dict[str, str]] = {
    "UN": {
        "fr": "ONU",
        "es": "ONU",
        "it": "ONU",
        "pt": "ONU",
        "ru": "ООН",
        "zh": "联合国",
        "ja": "国連",
        "ar": "الأمم المتحدة",
    },
    "EU": {
        "fr": "UE",
        "es": "UE",
        "it": "UE",
        "pt": "UE",
        "de": "EU",
        "ru": "ЕС",
        "zh": "欧盟",
        "ja": "EU",
    },
    "NATO": {
        "fr": "OTAN",
        "es": "OTAN",
        "it": "NATO",
        "pt": "OTAN",
        "ru": "НАТО",
        "zh": "北约",
    },
    "WHO": {
        "fr": "OMS",
        "es": "OMS",
        "it": "OMS",
        "pt": "OMS",
        "ru": "ВОЗ",
        "zh": "世卫组织",
        "ja": "WHO",
    },
    "IMF": {
        "fr": "FMI",
        "es": "FMI",
        "it": "FMI",
        "pt": "FMI",
        "ru": "МВФ",
        "zh": "国际货币基金组织",
    },
    "WTO": {
        "fr": "OMC",
        "es": "OMC",
        "it": "OMC",
        "pt": "OMC",
        "ru": "ВТО",
        "zh": "世贸组织",
    },
    "IAEA": {
        "fr": "AIEA",
        "es": "OIEA",
        "it": "AIEA",
        "ru": "МАГАТЭ",
        "zh": "国际原子能机构",
    },
    "ICJ": {
        "fr": "CIJ",
        "es": "CIJ",
        "ru": "МУС",
        "zh": "国际法院",
    },
    "USA": {
        "fr": "États-Unis",
        "es": "EEUU",
        "de": "USA",
        "it": "USA",
        "zh": "美国",
        "ja": "米国",
        "ru": "США",
    },
    "US": {
        "fr": "États-Unis",
        "es": "EEUU",
        "de": "USA",
        "it": "USA",
        "zh": "美国",
        "ja": "米国",
        "ru": "США",
    },
    "UK": {
        "fr": "Royaume-Uni",
        "es": "Reino Unido",
        "de": "Großbritannien",
        "zh": "英国",
        "ja": "英国",
        "ru": "Великобритания",
    },
}


def normalize_language_code(lang: str | None) -> str:
    cleaned = str(lang or "").strip().lower()
    return LANGUAGE_CODE_MAP.get(cleaned, cleaned[:2] if len(cleaned) >= 2 else "en")


def resolve_country_language(country: str | None, default: str = "en") -> str:
    code = str(country or "").strip().upper()
    return COUNTRY_DEFAULT_LANGUAGES.get(code, default)


def detect_text_language(text: str) -> str:
    sample = str(text or "")
    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", sample):
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff]", sample):
            return "ja"
        return "zh"
    if re.search(r"[\u0400-\u04ff]", sample):
        return "ru"
    if re.search(r"[\u0600-\u06ff]", sample):
        return "ar"
    if re.search(r"[\uac00-\ud7af]", sample):
        return "ko"
    lowered = sample.lower()
    words = set(re.findall(r"\b[a-zà-ÿ]{2,}\b", lowered))
    fr_markers = {"le", "la", "les", "des", "du", "un", "une", "est", "et", "dans", "sur", "pour", "avec"}
    de_markers = {"der", "die", "das", "und", "ist", "in", "den", "von", "mit", "für", "auf", "ein", "eine"}
    es_markers = {"el", "la", "los", "las", "un", "una", "en", "y", "del", "por", "con", "para", "es"}
    it_markers = {"il", "la", "lo", "i", "gli", "le", "un", "uno", "una", "ed", "del", "con", "per"}
    counts = {
        "fr": len(words.intersection(fr_markers)),
        "de": len(words.intersection(de_markers)),
        "es": len(words.intersection(es_markers)),
        "it": len(words.intersection(it_markers)),
    }
    best_lang, best_count = max(counts.items(), key=lambda item: item[1])
    if best_count >= 2:
        return best_lang
    return "en"


def align_query_with_assist(
    query: str,
    target_language: str,
    assist_lookup: Dict[str, Dict[str, str]] | None = None,
) -> str:
    """Fallback assistive entity replacement when an LLM call is not available."""
    target_lang = normalize_language_code(target_language)
    if not query or target_lang in {"en", ""}:
        return query

    lookup = assist_lookup or MULTILINGUAL_ASSIST_LOOKUP
    tokens = query.split()
    aligned: List[str] = []
    for token in tokens:
        clean = re.sub(r"[^\w]", "", token).upper()
        if clean in lookup and target_lang in lookup[clean]:
            aligned.append(lookup[clean][target_lang])
        else:
            aligned.append(token)
    return normalize_whitespace(" ".join(aligned)) or query


def align_queries_with_llm(
    ai_client: Any,
    queries: Sequence[str],
    target_languages: Sequence[str],
    *,
    story_context: str = "",
    assist_lookup: Dict[str, Dict[str, str]] | None = None,
) -> Dict[str, List[str]]:
    """Uses the language model to translate and localize news queries for any generic topic.

    The model can reference the assist lookup for official institutional entity translations,
    while freely localizing general news concepts (technology, finance, science, disasters, etc.).
    """
    cleaned_queries = [normalize_whitespace(str(q)) for q in queries if normalize_whitespace(str(q))]
    normalized_langs = {
        normalize_language_code(lang) for lang in target_languages
    } - {"en", ""}

    if not cleaned_queries or not normalized_langs:
        return {}

    lookup = assist_lookup or MULTILINGUAL_ASSIST_LOOKUP

    if ai_client is None:
        # Graceful fallback: assistive entity translation
        return {
            lang: [align_query_with_assist(q, lang, lookup) for q in cleaned_queries]
            for lang in normalized_langs
        }

    system_prompt = (
        "You localize news search queries into foreign languages for searching local native-language news feeds. "
        "Translate the core event concepts into natural, high-yield news search phrasing. "
        "Do not translate proper names of individuals, companies, or unique product brands unless a well-known localized name exists. "
        "Consult the assist lookup for standard international organizations when relevant. "
        "Return exactly one valid JSON object adhering to the schema."
    )

    user_payload = {
        "queries": cleaned_queries,
        "target_languages": sorted(normalized_langs),
        "assist_lookup": {
            acronym: {lang: trans for lang, trans in trans_map.items() if lang in normalized_langs}
            for acronym, trans_map in lookup.items()
            if any(lang in trans_map for lang in normalized_langs)
        },
        "story_context": story_context[:500] if story_context else "",
    }

    try:
        raw = ai_client.complete_json(
            system_prompt,
            compact_json(user_payload),
            label="perspectives cross-language query alignment",
            json_schema=MULTILINGUAL_QUERY_ALIGNMENT_SCHEMA,
        )
        if isinstance(raw, dict) and isinstance(raw.get("aligned_queries"), dict):
            result: Dict[str, List[str]] = {}
            for lang, raw_list in raw["aligned_queries"].items():
                norm_lang = normalize_language_code(lang)
                if isinstance(raw_list, list):
                    result[norm_lang] = [
                        normalize_whitespace(str(item))
                        for item in raw_list
                        if normalize_whitespace(str(item))
                    ]
            return result
    except Exception:
        pass

    # Fallback if AI call failed
    return {
        lang: [align_query_with_assist(q, lang, lookup) for q in cleaned_queries]
        for lang in normalized_langs
    }


def expand_multilingual_anchor_groups(
    anchor_groups: Sequence[Dict[str, Any]],
    target_languages: Sequence[str],
    assist_lookup: Dict[str, Dict[str, str]] | None = None,
) -> List[Dict[str, Any]]:
    normalized_langs = {normalize_language_code(lang) for lang in target_languages} - {"en"}
    if not normalized_langs:
        return [dict(group) for group in anchor_groups if isinstance(group, dict)]

    lookup = assist_lookup or MULTILINGUAL_ASSIST_LOOKUP
    expanded: List[Dict[str, Any]] = []
    for group in anchor_groups:
        if not isinstance(group, dict):
            continue
        current = dict(group)
        terms = [str(term).strip() for term in current.get("terms", []) if str(term).strip()]
        new_terms = list(terms)
        for term in terms:
            clean = re.sub(r"[^\w]", "", term).upper()
            if clean in lookup:
                for lang in normalized_langs:
                    if lang in lookup[clean]:
                        translated = lookup[clean][lang]
                        if translated and translated.lower() not in {t.lower() for t in new_terms}:
                            new_terms.append(translated)
        current["terms"] = new_terms
        expanded.append(current)
    return expanded
