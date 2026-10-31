"""Collect Google AI Overview answer candidates with auditable provenance.

An overview is generated text. Its references are citations, not verification of
the answer; every exported candidate remains explicitly unverified.
"""

import hashlib
import html
import re
from datetime import datetime, timezone
from urllib.parse import parse_qs, quote, quote_plus, urlparse


QA_EXTRACTION_MODES = ("legacy", "hybrid", "ai_overview")
AI_OVERVIEW_TIMEOUT_SECONDS = 30
_HTML_COMMENT = re.compile(
    r"(?:<|\\+x3[cC]|\\+u003[cC])!--.*?--(?:>|\\+x3[eE]|\\+u003[eE])", re.DOTALL,
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def _redact(value, api_key):
    if isinstance(value, dict):
        return {key: "[REDACTED]" if key == "api_key" else _redact(item, api_key)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, api_key) for item in value]
    if isinstance(value, str) and api_key:
        for secret in (api_key, quote(api_key, safe=""), quote_plus(api_key)):
            value = value.replace(secret, "[REDACTED]")
    return value


def _page_token(overview):
    if isinstance(overview.get("page_token"), str) and overview["page_token"].strip():
        return overview["page_token"]
    # Parse known SerpApi links, but never send credentials to a response-supplied URL.
    for field in ("serpapi_ai_overview_link", "serpapi_link"):
        link = overview.get(field)
        if not isinstance(link, str):
            continue
        try:
            parsed = urlparse(link)
        except ValueError:
            continue
        params = parse_qs(parsed.query)
        if (parsed.scheme == "https" and parsed.hostname == "serpapi.com"
                and params.get("engine") == ["google_ai_overview"]):
            return params.get("page_token", [None])[0]
    return None


def _overview_targets(response):
    # Related questions take priority within the per-seed request budget.
    for entry in response.get("related_questions") or []:
        if (isinstance(entry, dict) and entry.get("type") == "ai_overview"
                and isinstance(entry.get("question"), str) and entry["question"].strip()):
            yield entry["question"], "related_question", entry
    if isinstance(response.get("ai_overview"), dict):
        yield response["search_parameters"]["q"], "seed_query", response["ai_overview"]


def resolve_ai_overview_references(response, search_factory, api_key, max_requests=5):
    """Resolve deferred references immediately, before searching the next seed.

    Keep the original token/fields intact and attach the raw follow-up response
    under ``nativqa_ai_overview``. Failures are per overview, so classic answers
    and other questions from the same search are still retained. No automatic
    retry is made with an expired token.
    """
    if max_requests < 0:
        raise ValueError("max_requests must be nonnegative")
    requests_made = 0
    cache = {}
    for _, _, target in _overview_targets(response):
        overview = target.get("ai_overview")
        if not isinstance(overview, dict):
            overview = target
        state = {"retrieved_at_utc": utc_now()}
        if overview.get("text_blocks"):
            state["status"] = "embedded"
        else:
            token = _page_token(overview)
            if not token:
                state["status"] = "missing_token"
            elif token in cache:
                state = cache[token]
            elif requests_made >= max_requests:
                state["status"] = "request_limit"
            else:
                requests_made += 1
                try:
                    fetched = search_factory({
                        "engine": "google_ai_overview", "page_token": token,
                        "api_key": api_key,
                    }, timeout=AI_OVERVIEW_TIMEOUT_SECONDS).get_dict()
                    state["response"] = _redact(fetched, api_key)
                    if fetched.get("error"):
                        state.update(status="error", error=_redact(fetched["error"], api_key))
                    elif not fetched.get("ai_overview", {}).get("text_blocks"):
                        state.update(status="empty", error="No AI Overview text blocks returned")
                    else:
                        state["status"] = "fetched"
                except Exception as error:
                    state.update(status="error", error=_redact(str(error), api_key))
                state["retrieved_at_utc"] = utc_now()
                cache[token] = state
        target["nativqa_ai_overview"] = state
    return requests_made


def _clean_text(text):
    # Google can append literal/escaped HTML comments carrying internal metadata.
    # Preserve their original form in the raw response, but exclude them from QA.
    return " ".join(_HTML_COMMENT.sub("", html.unescape(text)).split())


def _flatten_blocks(nodes, inherited_indexes=()):
    """Preserve text order and citation indexes for nested lists/tables/sections."""
    if not isinstance(nodes, list):
        return
    for node in nodes:
        if isinstance(node, str):
            text = _clean_text(node)
            if text:
                yield {"text": text, "type": "paragraph",
                       "reference_indexes": list(inherited_indexes)}
            continue
        if not isinstance(node, dict):
            continue
        indexes = node.get("reference_indexes", inherited_indexes)
        if not isinstance(indexes, (list, tuple)):
            indexes = inherited_indexes
        indexes = [index for index in indexes if isinstance(index, int)]
        title = node.get("title", "")
        snippet = node.get("snippet", "")
        parts = [_clean_text(part) for part in (title, snippet) if isinstance(part, str)]
        text = ": ".join(part for part in parts if part)
        if text:
            yield {"text": text, "type": node.get("type", "paragraph"),
                   "reference_indexes": indexes}
        if isinstance(node.get("table"), list):
            for row in node["table"]:
                if isinstance(row, list):
                    cells = [_clean_text(str(cell)) for cell in row if isinstance(cell, (str, int, float))]
                    if cells:
                        yield {"text": " | ".join(cells), "type": "table",
                               "reference_indexes": indexes}
        for field in ("list", "text_blocks"):
            yield from _flatten_blocks(node.get(field), indexes)


def extract_ai_overview_answer(overview):
    """Extract an answer and citation mapping; return None without text or sources.

    This function performs no API calls and does not infer facts from reference
    titles/snippets. The answer comes only from the overview's text blocks.
    """
    if not isinstance(overview, dict):
        return None
    blocks = []
    for block in _flatten_blocks(overview.get("text_blocks", [])):
        if (blocks and block["type"] == blocks[-1]["type"] == "paragraph"
                and block["text"] == blocks[-1]["text"]):
            # A rendered paragraph and its metadata-bearing duplicate can coexist.
            blocks[-1]["reference_indexes"] = list(dict.fromkeys(
                blocks[-1]["reference_indexes"] + block["reference_indexes"]))
        else:
            blocks.append(block)
    if not blocks or not any(block["type"] != "heading" for block in blocks):
        return None
    references = overview.get("references") or []
    if not isinstance(references, list):
        return None
    indexes = {index for block in blocks for index in block["reference_indexes"]}
    sources = []
    for position, reference in enumerate(references):
        if not isinstance(reference, dict):
            continue
        index = reference.get("index", position)
        if not isinstance(index, int) or isinstance(index, bool):
            continue
        link = reference.get("link")
        if not isinstance(link, str):
            continue
        try:
            parsed = urlparse(link)
        except ValueError:
            continue
        if (parsed.scheme in ("http", "https") and parsed.netloc
                and (not indexes or index in indexes)):
            sources.append(reference)
    urls = list(dict.fromkeys(reference["link"] for reference in sources))
    if not urls:
        return None
    return {
        "answer": "\n".join(block["text"] for block in blocks),
        "source_urls": urls, "answer_blocks": blocks, "references": references,
        "citation_scope": "block" if indexes else "overview",
        "answer_origin": "google_ai_overview", "verification_status": "unverified",
        "gold_eligible": False,
    }


def iter_ai_overview_candidates(response):
    """Pair each overview with its own question, never with another question."""
    params = response["search_parameters"]
    for question, question_source, target in _overview_targets(response):
        state = target.get("nativqa_ai_overview", {})
        if state.get("status") in ("error", "empty", "missing_token", "request_limit"):
            continue
        if state.get("status") == "fetched":
            overview = state["response"]["ai_overview"]
        else:
            overview = target.get("ai_overview")
            if not isinstance(overview, dict):
                overview = target
        extracted = extract_ai_overview_answer(overview)
        if extracted:
            identifier = params["q"] + " " + question + " ai_overview"
            yield dict(
                extracted, data_id=hashlib.md5(identifier.encode()).hexdigest(),
                category=params["category"], input_query=params["q"], question=question,
                question_type="ai_overview", question_source=question_source,
                collected_at_utc=state.get("retrieved_at_utc") or response.get(
                    "nativqa_collection", {}).get("collected_at_utc"),
                search_parameters=params,
                search_metadata=response.get("search_metadata", {}),
                ai_overview_search_metadata=state.get("response", {}).get("search_metadata", {}),
            )
