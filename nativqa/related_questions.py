"""Bounded expansion of Google's People also ask results through SerpApi."""

import json
from collections import deque
from urllib.parse import parse_qs, urlparse

from .qa_extraction import _redact, utc_now


RELATED_QUESTIONS_TIMEOUT_SECONDS = 30


def validate_related_questions(engine, search_type, max_requests):
    if max_requests < 0:
        raise ValueError('related_questions_max_requests must be nonnegative')
    if max_requests and (engine.lower() != 'google' or search_type.lower() != 'text'):
        raise ValueError('google_related_questions expansion requires Google text search; '
                         'set --related_questions_max_requests 0 for other engines/modes')


def _next_page_token(entry):
    token = entry.get('next_page_token')
    if isinstance(token, str) and token.strip():
        return token
    link = entry.get('serpapi_link')
    if not isinstance(link, str):
        return None
    try:
        parsed = urlparse(link)
    except ValueError:
        return None
    params = parse_qs(parsed.query)
    if (parsed.scheme == 'https' and parsed.hostname == 'serpapi.com'
            and params.get('engine') == ['google_related_questions']):
        return params.get('next_page_token', [None])[0] or None
    return None


def _questions(value):
    if isinstance(value, list):
        for entry in value:
            if (isinstance(entry, dict) and isinstance(entry.get('question'), str)
                    and entry['question'].strip()):
                yield entry


def _identity(entry):
    # Keep different answers/overviews to the same question, while ignoring
    # changing expansion tokens, logos, and request provenance.
    return json.dumps({key: entry[key] for key in (
        'question', 'type', 'snippet', 'list', 'link', 'text_blocks', 'references',
        'ai_overview', 'page_token',
    ) if key in entry}, sort_keys=True, ensure_ascii=False)


def expand_related_questions(response, search_factory, api_key, max_requests=0,
                             on_questions=None):
    """Append expanded questions and retain raw follow-ups under collection metadata.

    Traverse tokens breadth first, once per distinct token per seed, including
    failed requests in the budget. ``on_questions`` can resolve fresh AI Overview
    tokens in each batch before the next expansion request. All requests use the
    fixed SerpApi client endpoint, never a response-supplied URL.
    """
    if max_requests < 0:
        raise ValueError('related_questions_max_requests must be nonnegative')
    if max_requests == 0:
        return 0
    params = response.get('search_parameters', {})
    validate_related_questions(params.get('engine', 'google'), 'text', max_requests)
    questions = list(_questions(response.get('related_questions')))
    if questions:
        response['related_questions'] = questions
    else:
        response.pop('related_questions', None)
    queue = deque()
    seen_tokens = set()
    seen_questions = {_identity(entry) for entry in questions}
    collection = {
        'engine': 'google_related_questions', 'max_requests': max_requests,
        'requests_made': 0, 'added_questions': 0, 'requests': [],
    }
    response['nativqa_related_questions'] = collection

    def enqueue(entries, depth):
        for entry in entries:
            token = _next_page_token(entry)
            if token and token not in seen_tokens:
                seen_tokens.add(token)
                queue.append((token, entry['question'], depth))

    enqueue(questions, 1)
    while queue and collection['requests_made'] < max_requests:
        token, parent_question, depth = queue.popleft()
        collection['requests_made'] += 1
        state = {'parent_question': parent_question, 'next_page_token': token,
                 'depth': depth, 'retrieved_at_utc': utc_now()}
        added = []
        try:
            search = search_factory({
                'engine': 'google_related_questions', 'next_page_token': token,
                'api_key': api_key,
            })
            search.timeout = RELATED_QUESTIONS_TIMEOUT_SECONDS
            fetched = search.get_dict()
            if not isinstance(fetched, dict):
                raise ValueError('Expected a JSON object from google_related_questions')
            state['response'] = _redact(fetched, api_key)
            if fetched.get('error'):
                state.update(status='error', error=_redact(fetched['error'], api_key))
            else:
                batch = list(_questions(fetched.get('related_questions')))
                state['status'] = 'fetched' if batch else 'empty'
                # Discover tokens even when an answer is already present locally.
                enqueue(batch, depth + 1)
                for entry in batch:
                    identity = _identity(entry)
                    if identity in seen_questions:
                        continue
                    seen_questions.add(identity)
                    candidate = _redact(entry, api_key)
                    candidate['nativqa_related_question'] = {
                        'engine': 'google_related_questions', 'parent_question': parent_question,
                        'depth': depth, 'request_index': collection['requests_made'],
                        'search_metadata': _redact(fetched.get('search_metadata', {}), api_key),
                    }
                    questions.append(candidate)
                    added.append(candidate)
        except Exception as error:
            state.update(status='error', error=_redact(str(error), api_key))
        state['retrieved_at_utc'] = utc_now()
        collection['requests'].append(state)
        collection['added_questions'] += len(added)
        if added and on_questions is not None:
            on_questions(added)
    collection['pending_tokens'] = len(queue)
    collection['request_limit_reached'] = bool(queue)
    return collection['requests_made']
