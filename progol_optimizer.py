import numpy as np
import pandas as pd

def calculate_ticket_cost(doubles, triples):
    return (2**doubles * 3**triples) * 15

def get_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-9))

def optimize_progol_ticket(match_probs, budget=2000):
    """Optimizes budget for any number of matches."""
    match_stats = []
    for i, p in enumerate(match_probs):
        entropy = get_entropy(p)
        match_stats.append({
            'id': i, 'probs': p, 'entropy': entropy
        })
    
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    
    # Configuration search
    max_d, max_t = 0, 0
    for t in range(10): # Allow more triples for larger slates
        for d in range(14): 
            cost = calculate_ticket_cost(d, t)
            if cost <= budget:
                if (d + t) <= len(match_probs):
                    if cost > calculate_ticket_cost(max_d, max_t):
                        max_d, max_t = d, t

    # DYNAMIC CONFIG SIZE
    config = ['S'] * len(match_probs)
    sorted_indices = df.index.tolist()
    
    # Apply Triples and Doubles to most uncertain matches
    for i in range(min(max_t, len(sorted_indices))):
        config[sorted_indices[i]] = 'T'
    for i in range(max_t, min(max_t + max_d, len(sorted_indices))):
        config[sorted_indices[i]] = 'D'
        
    return config, calculate_ticket_cost(max_d, max_t), max_d, max_t

def print_final_ticket(match_ids, probs, config):
    print("\n" + "="*80)
    print(f"{'GAME':<5} | {'MATCHUP':<35} | {'TYPE':<5} | {'ADJUSTED WIN PROB'}")
    print("-" * 80)
    
    total_ticket_prob = 1.0
    for i, (mid, p, c) in enumerate(zip(match_ids, probs, config)):
        best_idx = np.argmax(p)
        label = {0:'L', 1:'E', 2:'V'}[best_idx]
        
        if c == 'S': display, prob_cont = label, p[best_idx]
        elif c == 'D':
            top2 = np.argsort(p)[-2:]
            display = "/".join([{0:'L', 1:'E', 2:'V'}[x] for x in sorted(top2)])
            prob_cont = np.sum(p[top2])
        else: display, prob_cont = "L/E/V", 1.0
            
        total_ticket_prob *= prob_cont
        print(f"{i+1:2}    | {str(mid):<35} | {c:<5} | {prob_cont*100:6.1f}%")
    
    print("-" * 80)
    print(f"ESTIMATED TICKET HIT PROBABILITY: {total_ticket_prob * 100:.6f}%")
    print("="*80)
