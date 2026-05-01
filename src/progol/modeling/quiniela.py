"""Quiniela diversification + budget optimizer.

Two responsibilities:

1. `top_n_quinielas`: given per-match probabilities (p_L, p_E, p_V), return the N most
   joint-probable single-bet quinielas, sorted descending. Reasoning: a player who buys
   the same ticket many times wastes money; if they buy the *next* most likely ticket,
   they cover an alternate scenario without wasting any cell.

2. `optimize_budget`: given a budget, base ticket cost, and per-match probabilities,
   pick which matches to upgrade to double / triple so the *coverage probability*
   (probability the played ticket covers the actual outcome) is maximized within budget.
   Cost rule (Progol oficial): cost = base * 2**doubles * 3**triples.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

LABELS = ('L', 'E', 'V')
BASE_COST_MXN = 15  # Progol estándar (loterianacional.gob.mx)


# ---------------------------------------------------------------------------
# Top-N
# ---------------------------------------------------------------------------

def top_n_quinielas(probs: np.ndarray, n: int = 10) -> list[dict]:
    """Return the n most joint-probable quinielas.

    `probs` is a (M, 3) array where columns are [P(L), P(E), P(V)] for each match.
    Uses a best-first search over quiniela states. With n small (<= 50) this is fast even
    for M = 21 matches.
    """
    n_matches = probs.shape[0]
    sort_idx = np.argsort(-probs, axis=1)  # per-match outcome ranking

    # Each state: (neg_log_prob, [picked_outcome_per_match]).
    log_probs = np.log(np.clip(probs, 1e-9, 1.0))

    # Start with the argmax pick for every match.
    initial_picks = sort_idx[:, 0].tolist()
    initial_score = -sum(log_probs[i, initial_picks[i]] for i in range(n_matches))

    seen: set[tuple[int, ...]] = {tuple(initial_picks)}
    heap: list[tuple[float, list[int]]] = [(initial_score, initial_picks)]
    results: list[dict] = []

    while heap and len(results) < n:
        neg_logp, picks = heapq.heappop(heap)
        results.append({
            'quiniela': [LABELS[p] for p in picks],
            'log_prob': -neg_logp,
            'joint_prob': float(np.exp(-neg_logp)),
        })

        # Generate neighbors by swapping one match to its next-best outcome.
        # neg_logp = -sum(log_probs[i, picks[i]]); swapping picks[m] -> new_outcome:
        # new_neg_logp = neg_logp + log_probs[m, picks[m]] - log_probs[m, new_outcome]
        for m in range(n_matches):
            current_rank = list(sort_idx[m]).index(picks[m])
            if current_rank < 2:
                new_outcome = int(sort_idx[m, current_rank + 1])
                new_picks = picks[:m] + [new_outcome] + picks[m + 1:]
                key = tuple(new_picks)
                if key in seen:
                    continue
                seen.add(key)
                new_neg_logp = neg_logp + log_probs[m, picks[m]] - log_probs[m, new_outcome]
                heapq.heappush(heap, (new_neg_logp, new_picks))

    return results


# ---------------------------------------------------------------------------
# Budget optimizer
# ---------------------------------------------------------------------------

@dataclass
class BudgetPlan:
    coverage_prob: float
    cost: float
    n_tickets: int
    doubles: list[int]
    triples: list[int]
    base_picks: list[str]
    matches_summary: list[dict]


def _coverage_probability(probs: np.ndarray, base_picks: list[int],
                          doubles: list[int], triples: list[int]) -> float:
    """Probability that the played combination *covers* the actual outcome.

    For each match, the played outcomes are:
      - triple: all three
      - double: top-2 (by probability)
      - else:   top-1 (the base pick)
    Coverage prob is the product over all matches of the sum of P() over played outcomes.
    """
    p = 1.0
    for m in range(probs.shape[0]):
        if m in triples:
            p *= 1.0  # all 3 outcomes covered
        elif m in doubles:
            top2 = np.argsort(-probs[m])[:2]
            p *= float(probs[m, top2].sum())
        else:
            p *= float(probs[m, base_picks[m]])
    return p


def _cost(n_doubles: int, n_triples: int, base_cost: float = BASE_COST_MXN) -> float:
    return base_cost * (2 ** n_doubles) * (3 ** n_triples)


def optimize_budget(probs: np.ndarray, budget: float,
                    base_cost: float = BASE_COST_MXN,
                    max_doubles: int = 6, max_triples: int = 4,
                    only_first_n: int = 14) -> BudgetPlan:
    """Pick which of the first `only_first_n` matches (Progol main = 14) to play as
    double / triple to maximize coverage probability under the budget cap.

    Returns the optimal plan. Brute-forces over (n_doubles, n_triples) feasible counts
    given the budget; for each count picks the matches with the lowest top-1 probability
    (i.e., where doubling/tripling adds the most coverage)."""
    M = min(only_first_n, probs.shape[0])
    p = probs[:M]
    base_picks = np.argmax(p, axis=1).tolist()
    sorted_p = -np.sort(-p, axis=1)  # descending per row

    # Marginal gain ranking: for each match, gain from upgrading top1 -> double -> triple.
    # Doubling adds P(2nd best); tripling on top of double adds P(3rd best).
    gain_double = sorted_p[:, 1] / np.clip(sorted_p[:, 0], 1e-9, 1.0)  # multiplicative gain
    # We want to pick the matches where the doubled coverage / single coverage ratio is highest,
    # i.e., where the model is least confident.
    double_priority = np.argsort(-gain_double).tolist()
    triple_priority = np.argsort(-sorted_p[:, 2]).tolist()

    best = BudgetPlan(coverage_prob=0.0, cost=base_cost, n_tickets=1,
                      doubles=[], triples=[],
                      base_picks=[LABELS[i] for i in base_picks],
                      matches_summary=[])

    # Try every combination of (#triples, #doubles) within budget.
    for nt in range(min(max_triples, M) + 1):
        for nd in range(min(max_doubles, M - nt) + 1):
            cost = _cost(nd, nt, base_cost)
            if cost > budget:
                continue
            # Pick which matches: triples take priority of the lowest-confidence; then
            # doubles fill up next.
            triple_picks = triple_priority[:nt]
            remaining = [m for m in double_priority if m not in triple_picks]
            double_picks = remaining[:nd]
            cov = _coverage_probability(p, base_picks, double_picks, triple_picks)
            if cov > best.coverage_prob:
                best = BudgetPlan(
                    coverage_prob=cov,
                    cost=cost,
                    n_tickets=2 ** nd * 3 ** nt,
                    doubles=double_picks,
                    triples=triple_picks,
                    base_picks=[LABELS[i] for i in base_picks],
                    matches_summary=[],
                )

    # Build a per-match summary on the best plan.
    summary = []
    for m in range(M):
        if m in best.triples:
            kind, played = 'triple', list(LABELS)
        elif m in best.doubles:
            top2 = np.argsort(-p[m])[:2]
            kind, played = 'double', [LABELS[i] for i in top2]
        else:
            kind, played = 'single', [LABELS[best_picks_m] for best_picks_m in [base_picks[m]]]
        summary.append({
            'match_index': m + 1,
            'kind': kind,
            'played': played,
            'probs': {'L': float(p[m, 0]), 'E': float(p[m, 1]), 'V': float(p[m, 2])},
        })
    best.matches_summary = summary
    return best


# ---------------------------------------------------------------------------
# Pretty-printers (for CLI / chat)
# ---------------------------------------------------------------------------

def format_top_n(top: list[dict], match_names: list[str] | None = None) -> str:
    lines = []
    n_matches = len(top[0]['quiniela'])
    if match_names is None:
        match_names = [f"M{i+1}" for i in range(n_matches)]
    header = "RANK | " + " | ".join(f"{i+1:>2}" for i in range(n_matches)) + " | JOINT-PROB"
    lines.append(header)
    lines.append("-" * len(header))
    for rank, q in enumerate(top, 1):
        row = f"{rank:>4} | " + " | ".join(f"{r:>2}" for r in q['quiniela']) + f" | {q['joint_prob']*100:6.4f}%"
        lines.append(row)
    return "\n".join(lines)


def format_plan(plan: BudgetPlan) -> str:
    lines = [
        f"Plan optimizado:",
        f"  Costo:       ${plan.cost:.2f} MXN  ({plan.n_tickets} boletos cubiertos)",
        f"  Cobertura:   {plan.coverage_prob*100:.4f}%  (prob. de pegarle a los 14)",
        f"  Triples ({len(plan.triples)}): {sorted(m+1 for m in plan.triples)}",
        f"  Dobles  ({len(plan.doubles)}): {sorted(m+1 for m in plan.doubles)}",
        "",
        "Por partido:",
    ]
    for s in plan.matches_summary:
        flag = {'triple': '[T]', 'double': '[D]', 'single': '   '}[s['kind']]
        played = "/".join(s['played'])
        probs = s['probs']
        lines.append(f"  {flag} #{s['match_index']:>2}  jugar {played:>5}   "
                     f"L:{probs['L']*100:5.1f}% E:{probs['E']*100:5.1f}% V:{probs['V']*100:5.1f}%")
    return "\n".join(lines)
