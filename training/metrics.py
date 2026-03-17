from typing import List
import math

import nltk
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from nltk.translate.meteor_score import meteor_score


def _tokenize_texts(texts: List[str]) -> List[List[str]]:
    return [t.split() if t else [] for t in texts]


def bleu_scores(references: List[str], predictions: List[str]):
    ref_tokens = [[r.split()] for r in references]
    pred_tokens = [p.split() for p in predictions]
    smoothie = SmoothingFunction().method4

    bleu1 = corpus_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
    bleu4 = corpus_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    return bleu1, bleu4


def _lcs(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


def rouge_l(references: List[str], predictions: List[str]) -> float:
    scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        lcs = _lcs(ref_tokens, pred_tokens)
        if lcs == 0:
            scores.append(0.0)
            continue
        precision = lcs / max(1, len(pred_tokens))
        recall = lcs / max(1, len(ref_tokens))
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append((2 * precision * recall) / (precision + recall))
    return sum(scores) / max(1, len(scores))


def meteor(references: List[str], predictions: List[str]) -> float:
    scores = []
    try:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
    except Exception:
        pass

    for ref, pred in zip(references, predictions):
        try:
            scores.append(meteor_score([ref], pred))
        except Exception:
            scores.append(0.0)
    return sum(scores) / max(1, len(scores))


def exact_match(references: List[str], predictions: List[str]) -> float:
    matches = [1.0 if r == p else 0.0 for r, p in zip(references, predictions)]
    return sum(matches) / max(1, len(matches))


def compute_metrics(references: List[str], predictions: List[str]) -> dict:
    bleu1, bleu4 = bleu_scores(references, predictions)
    rouge = rouge_l(references, predictions)
    meteor_score = meteor(references, predictions)
    em = exact_match(references, predictions)
    return {
        "bleu1": bleu1,
        "bleu4": bleu4,
        "rouge_l": rouge,
        "meteor": meteor_score,
        "exact_match": em,
    }
