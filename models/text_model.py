"""
text_model.py
Uses trained TF-IDF + Logistic Regression classifier (text_classifier.pkl)
to analyse journal entries. Falls back to keyword engine if model not found.
Models are lazy-loaded (cached after first request) to save RAM on startup.
"""

import os
import re
import pickle
import numpy as np

MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'text_classifier.pkl')
TFIDF_PATH  = os.path.join(os.path.dirname(__file__), 'tfidf_vectorizer.pkl')

CLASS_RISK = {
    'Normal':               5,
    'Anxiety':             55,
    'Stress':              45,
    'Depression':          70,
    'Bipolar':             65,
    'Personality disorder':60,
    'Suicidal':            95,
}

CLASS_SENTIMENT = {
    'Normal':               'Positive / Neutral',
    'Anxiety':              'Moderately Negative',
    'Stress':               'Mildly Negative',
    'Depression':           'Severely Negative',
    'Bipolar':              'Moderately Negative',
    'Personality disorder': 'Moderately Negative',
    'Suicidal':             'Severely Negative',
}


# ── Lazy-loaded model cache ───────────────────────────────────────────────────

_clf   = None
_tfidf = None

def _load_text_models():
    global _clf, _tfidf
    if _clf is None:
        with open(MODEL_PATH, 'rb') as f: _clf   = pickle.load(f)
        with open(TFIDF_PATH, 'rb') as f: _tfidf = pickle.load(f)


# ── ML inference ──────────────────────────────────────────────────────────────

def _ml_analyze(text):
    _load_text_models()
    clf   = _clf
    tfidf = _tfidf

    vec        = tfidf.transform([text])
    label      = clf.predict(vec)[0]
    proba_arr  = clf.predict_proba(vec)[0]
    proba      = dict(zip(clf.classes_, proba_arr.tolist()))

    depression_score = round((proba.get('Depression', 0) +
                               proba.get('Suicidal',   0) * 0.8 +
                               proba.get('Bipolar',    0) * 0.5) * 100, 1)

    anxiety_score    = round((proba.get('Anxiety',               0) +
                               proba.get('Personality disorder', 0) * 0.4) * 100, 1)

    stress_score     = round(proba.get('Stress', 0) * 100, 1)

    risk_score = CLASS_RISK.get(label, 30)
    sentiment  = CLASS_SENTIMENT.get(label, 'Mildly Negative')

    scores_dict = {
        'Depression': depression_score,
        'Anxiety':    anxiety_score,
        'Stress':     stress_score,
    }
    dominant = max(scores_dict, key=scores_dict.get) if risk_score > 10 else 'None detected'

    return {
        'method':             'ML Model (TF-IDF + Logistic Regression)',
        'risk_score':         risk_score,
        'sentiment':          sentiment,
        'ml_label':           label,
        'ml_confidence':      round(max(proba_arr) * 100, 1),
        'dominant_condition': dominant,
        'condition_scores':   {k: round(v, 1) for k, v in scores_dict.items()},
        'class_probabilities':{k: round(v * 100, 1) for k, v in proba.items()},
        'word_count':         len(text.split()),
    }


# ── Keyword fallback ──────────────────────────────────────────────────────────

DEPRESSION_KEYWORDS = [
    'sad', 'hopeless', 'worthless', 'empty', 'numb', 'tired', 'exhausted',
    'lonely', 'alone', 'isolated', 'crying', 'cried', 'tears', 'depressed',
    'depression', 'miserable', 'unhappy', 'hate myself', 'no point',
    'give up', 'pointless', 'meaningless', 'dark', 'helpless', 'lost',
    'broken', 'dead inside', 'no motivation', 'cant sleep', "can't sleep",
    'sleeping too much', 'no appetite', 'lost appetite'
]
ANXIETY_KEYWORDS = [
    'anxious', 'anxiety', 'worried', 'worry', 'nervous', 'panic', 'fear',
    'scared', 'terrified', 'overwhelmed', 'stressed', 'stress', 'tension',
    'racing heart', 'cant breathe', "can't breathe", 'overthinking',
    'restless', 'uneasy', 'dread', 'terrifying', 'phobia', 'shaking',
    'trembling', 'sweating', 'heart pounding', 'catastrophe'
]
STRESS_KEYWORDS = [
    'pressure', 'deadline', 'overwhelmed', 'too much', 'burden', 'struggle',
    'exhausted', 'burned out', 'burnout', 'overworked', 'frustrated',
    'irritable', 'angry', 'cant cope', "can't cope", 'no time', 'failing',
    'failure', 'behind', 'behind schedule', 'expectations'
]
POSITIVE_KEYWORDS = [
    'happy', 'joyful', 'great', 'wonderful', 'excited', 'grateful',
    'thankful', 'hopeful', 'optimistic', 'motivated', 'energetic',
    'peaceful', 'calm', 'relaxed', 'content', 'good', 'fine', 'better'
]
NEGATIONS = ['not', "don't", "doesn't", "didn't", "never", "no", "hardly", "barely"]


def _check_negation(words, idx):
    context = words[max(0, idx - 3):idx]
    return any(neg in context for neg in NEGATIONS)


def _keyword_analyze(text):
    text_lower = text.lower()
    words      = re.findall(r"\b\w+(?:'\w+)?\b", text_lower)
    total      = max(len(words), 1)

    dc = ac = sc = pc = 0
    detected = {'depression': [], 'anxiety': [], 'stress': [], 'positive': []}

    for i, w in enumerate(words):
        neg = _check_negation(words, i)
        if w in [k.split()[0] for k in DEPRESSION_KEYWORDS if ' ' not in k]:
            if not neg: dc += 1; detected['depression'].append(w)
        if w in [k.split()[0] for k in ANXIETY_KEYWORDS if ' ' not in k]:
            if not neg: ac += 1; detected['anxiety'].append(w)
        if w in [k.split()[0] for k in STRESS_KEYWORDS if ' ' not in k]:
            if not neg: sc += 1; detected['stress'].append(w)
        if w in POSITIVE_KEYWORDS:
            pc += 1; detected['positive'].append(w)

    for phrase in DEPRESSION_KEYWORDS + ANXIETY_KEYWORDS + STRESS_KEYWORDS:
        if ' ' in phrase and phrase in text_lower:
            if phrase in DEPRESSION_KEYWORDS: dc += 2
            elif phrase in ANXIETY_KEYWORDS:  ac += 2
            else:                             sc += 2

    norm = 100 / total
    ds   = min(dc * norm * 8, 100)
    as_  = min(ac * norm * 8, 100)
    ss   = min(sc * norm * 8, 100)
    pos  = min(pc * norm * 5, 30)

    raw   = ds * 0.4 + as_ * 0.35 + ss * 0.25
    score = max(0, raw - pos)

    dominant = max({'Depression': ds, 'Anxiety': as_, 'Stress': ss},
                   key=lambda k: {'Depression': ds, 'Anxiety': as_, 'Stress': ss}[k])

    if score < 20:   sentiment = 'Positive / Neutral'
    elif score < 45: sentiment = 'Mildly Negative'
    elif score < 70: sentiment = 'Moderately Negative'
    else:            sentiment = 'Severely Negative'

    return {
        'method':            'Keyword Engine (fallback)',
        'risk_score':        round(score, 1),
        'sentiment':         sentiment,
        'dominant_condition': dominant if score > 15 else 'None detected',
        'condition_scores':  {'Depression': round(ds,1), 'Anxiety': round(as_,1), 'Stress': round(ss,1)},
        'detected_keywords': {k: list(set(v)) for k, v in detected.items() if v},
        'word_count':        total,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def analyze_text(text):
    if os.path.exists(MODEL_PATH) and os.path.exists(TFIDF_PATH):
        try:
            return _ml_analyze(text)
        except Exception as e:
            pass  # fall through to keyword engine
    return _keyword_analyze(text)
