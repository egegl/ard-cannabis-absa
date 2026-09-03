"""Acceptance checks for generated posts: aspect present, length, exact and near duplicates."""

import re

import config as cfg

ASPECT_MISSING = "ASPECT_MISSING"
LENGTH_VIOLATION = "LENGTH_VIOLATION"
EXACT_DUPLICATE = "EXACT_DUPLICATE"
NEAR_DUPLICATE = "NEAR_DUPLICATE"

_WS = re.compile(r"\s+")


def normalize(text):
    return _WS.sub(" ", str(text)).casefold().strip()


def char_ngrams(text, n=cfg.NEAR_DUP_NGRAM):
    if len(text) < n:
        return frozenset((text,)) if text else frozenset()
    return frozenset(text[i:i + n] for i in range(len(text) - n + 1))


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class ReferenceSet:
    """real training texts plus every accepted post so far"""

    def __init__(self, train_texts):
        self._exact = set()
        self._ngrams = []
        for t in train_texts:
            self.add(t)

    def add(self, text):
        norm = normalize(text)
        self._exact.add(norm)
        self._ngrams.append(char_ngrams(norm))

    def near_duplicate(self, normalized_text, threshold=cfg.NEAR_DUP_JACCARD):
        grams = char_ngrams(normalized_text)
        n = len(grams)
        for ref in self._ngrams:
            m = len(ref)
            if n == 0 or m == 0 or min(n, m) / max(n, m) < threshold:  # size bound
                continue
            j = jaccard(grams, ref)
            if j >= threshold:
                return True, j
        return False, 0.0

    def contains_exact(self, normalized_text):
        return normalized_text in self._exact


def check_candidate(post_text, aspect, reference):
    """(None, None) if accepted else (reason, detail)"""
    norm_text = normalize(post_text)
    norm_aspect = normalize(aspect)

    if not norm_aspect or len(norm_aspect) > cfg.ASPECT_MAX_CHARS:
        return ASPECT_MISSING, f"aspect empty or longer than {cfg.ASPECT_MAX_CHARS} chars"
    if norm_aspect not in norm_text:
        return ASPECT_MISSING, f"aspect {norm_aspect!r} not found in post text"

    n_chars = len(post_text.strip())
    if n_chars < cfg.POST_MIN_CHARS or n_chars > cfg.POST_MAX_CHARS:
        return LENGTH_VIOLATION, f"{n_chars} chars outside [{cfg.POST_MIN_CHARS}, {cfg.POST_MAX_CHARS}]"

    if reference.contains_exact(norm_text):
        return EXACT_DUPLICATE, "normalized text equals a reference text"

    is_dup, j = reference.near_duplicate(norm_text)
    if is_dup:
        return NEAR_DUPLICATE, f"char-{cfg.NEAR_DUP_NGRAM}-gram jaccard {j:.4f} >= {cfg.NEAR_DUP_JACCARD}"

    return None, None
