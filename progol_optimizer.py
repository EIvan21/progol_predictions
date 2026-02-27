import numpy as np
import pandas as pd

def calculate_ticket_cost(doubles, triples):
    return (2**doubles * 3**triples) * 15

def get_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-9))

def optimize_progol_ticket(match_probs, budget=2000):
    match_stats = []
    for i, p in enumerate(match_probs):
        entropy = get_entropy(p)
        best_idx = np.argmax(p)
        match_stats.append({
            'id': i, 'probs': p, 'entropy': entropy,
            'best_single': {0:'L', 1:'E', 2:'V'}[best_idx]
        })
    
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    doubles, triples = 0, 0
    max_d, max_t = 0, 0
    
    for t in range(6):
        for d in range(9):
            cost = calculate_ticket_cost(d, t)
            if cost <= budget and (d + t) <= 14:
                if cost > calculate_ticket_cost(max_d, max_t):
                    max_d, max_t = d, t

    config = ['S'] * 14
    sorted_indices = df.index.tolist()
    for i in range(max_t): config[sorted_indices[i]] = 'T'
    for i in range(max_t, max_t + max_d): config[sorted_indices[i]] = 'D'
        
    return config, calculate_ticket_cost(max_d, max_t), max_d, max_t

def print_final_ticket(match_ids, probs, config):
    print("\n" + "="*65)
    print(f"{'GAME':<5} | {'MATCH ID':<10} | {'PREDICTION':<10} | {'TYPE':<5} | {'WIN PROB':<8}")
    print("-" * 65)
    
    total_ticket_prob = 1.0
    for i, (mid, p, c) in enumerate(zip(match_ids, probs, config)):
        best_idx = np.argmax(p)
        label = {0:'L', 1:'E', 2:'V'}[best_idx]
        
        if c == 'S': 
            display, prob_contribution = label, p[best_idx]
        elif c == 'D':
            top2 = np.argsort(p)[-2:]
            display = "/".join([{0:'L', 1:'E', 2:'V'}[x] for x in sorted(top2)])
            prob_contribution = np.sum(p[top2])
        else:
            display, prob_contribution = "L/E/V", 1.0
            
        total_ticket_prob *= prob_contribution
        print(f"{i+1:2}    | {mid:<10} | {display:<10} | {c:<5} | {prob_contribution*100:6.1f}%")
    
    print("-" * 65)
    print(f"ESTIMATED TICKET HIT PROBABILITY: {total_ticket_prob * 100:.6f}%")
    print(f"EXPECTED SUCCESS RATE: {total_ticket_prob * 14:.2f} games")
    print("="*65)
