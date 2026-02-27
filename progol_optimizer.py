import numpy as np
import pandas as pd

def calculate_ticket_cost(doubles, triples):
    """Progol Mexico Cost Formula."""
    return (2**doubles * 3**triples) * 15

def get_entropy(probs):
    """Calculates Shannon Entropy. Higher entropy = more uncertainty."""
    return -np.sum(probs * np.log2(probs + 1e-9))

def optimize_progol_ticket(match_probs, budget=2000):
    """
    Finds the best configuration of S, D, T within budget.
    match_probs: List of 14 probability arrays [P(L), P(E), P(V)]
    """
    # 1. Rank matches by Uncertainty (Entropy)
    match_stats = []
    for i, p in enumerate(match_probs):
        entropy = get_entropy(p)
        best_idx = np.argmax(p)
        top_two_sum = np.sum(np.sort(p)[-2:])
        match_stats.append({
            'id': i,
            'probs': p,
            'entropy': entropy,
            'top_two_prob': top_two_sum,
            'best_single': {0:'L', 1:'E', 2:'V'}[best_idx]
        })
    
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    
    # 2. Iterative Budget Allocation
    # Start with all Singles
    doubles = 0
    triples = 0
    config = ['S'] * 14
    
    # Simple Greedy Optimization:
    # Upgrade highest entropy games to triples first, then doubles
    # until we hit the $2,000 limit.
    
    # Let's try different combinations of Doubles and Triples
    # Example constraints for Progol: max 8 doubles or 5 triples usually.
    best_config = config.copy()
    max_d, max_t = 0, 0
    
    for t in range(6): # Try up to 5 triples
        for d in range(9): # Try up to 8 doubles
            cost = calculate_ticket_cost(d, t)
            if cost <= budget:
                if (d + t) <= 14:
                    if cost > calculate_ticket_cost(max_d, max_t):
                        max_d, max_t = d, t

    # 3. Assign Doubles and Triples to the most uncertain matches
    sorted_indices = df.index.tolist()
    for i in range(max_t):
        config[sorted_indices[i]] = 'T'
    for i in range(max_t, max_t + max_d):
        config[sorted_indices[i]] = 'D'
        
    return config, calculate_ticket_cost(max_d, max_t), max_d, max_t

def print_final_ticket(match_ids, probs, config):
    print("
" + "="*65)
    print(f"{'GAME':<5} | {'MATCH ID':<10} | {'PREDICTION':<10} | {'TYPE':<5} | {'WIN PROB':<8}")
    print("-" * 65)
    
    total_ticket_prob = 1.0
    for i, (mid, p, c) in enumerate(zip(match_ids, probs, config)):
        best_idx = np.argmax(p)
        label = {0:'L', 1:'E', 2:'V'}[best_idx]
        
        if c == 'S': 
            display = label
            prob_contribution = p[best_idx]
        elif c == 'D':
            # Pick top 2
            top2 = np.argsort(p)[-2:]
            display = "/".join([{0:'L', 1:'E', 2:'V'}[x] for x in sorted(top2)])
            prob_contribution = np.sum(p[top2])
        else:
            display = "L/E/V"
            prob_contribution = 1.0
            
        total_ticket_prob *= prob_contribution
        print(f"{i+1:2}    | {mid:<10} | {display:<10} | {c:<5} | {prob_contribution*100:6.1f}%")
    
    print("-" * 65)
    print(f"ESTIMATED TICKET HIT PROBABILITY: {total_ticket_prob * 100:.6f}%")
    print(f"EXPECTED SUCCESS RATE: {total_ticket_prob * 100 * 14:.2f} games")
    print("="*65)

if __name__ == "__main__":
    # Test with mock data
    mock_probs = [np.random.dirichlet(np.ones(3)) for _ in range(14)]
    config, cost, d, t = optimize_progol_ticket(mock_probs)
    print(f"Optimal Config: {d} Doubles, {t} Triples. Total Cost: ${cost} MXN")
    print(f"Structure: {config}")
