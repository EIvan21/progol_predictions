import numpy as np
import pandas as pd

def calculate_ticket_cost(doubles, triples):
    return (2**doubles * 3**triples) * 15

def get_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-9))

def get_custom_ticket_config(match_probs, num_doubles, num_triples):
    """Generates a config based on exact counts requested by user."""
    match_stats = []
    for i, p in enumerate(match_probs):
        match_stats.append({'id': i, 'entropy': get_entropy(p)})
    
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    sorted_indices = df.index.tolist()
    
    config = ['S'] * len(match_probs)
    # Assign Triples to top most uncertain
    for i in range(min(num_triples, len(sorted_indices))):
        config[sorted_indices[i]] = 'T'
    # Assign Doubles to next most uncertain
    for i in range(num_triples, min(num_triples + num_doubles, len(sorted_indices))):
        config[sorted_indices[i]] = 'D'
        
    return config, calculate_ticket_cost(num_doubles, num_triples)

def optimize_progol_ticket(match_probs, budget=2000):
    """Automatic greedy optimization based on budget."""
    match_stats = []
    for i, p in enumerate(match_probs):
        match_stats.append({'id': i, 'entropy': get_entropy(p)})
    
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    
    max_d, max_t = 0, 0
    for t in range(10):
        for d in range(14): 
            cost = calculate_ticket_cost(d, t)
            if cost <= budget and (d + t) <= len(match_probs):
                if cost > calculate_ticket_cost(max_d, max_t):
                    max_d, max_t = d, t

    config = ['S'] * len(match_probs)
    sorted_indices = df.index.tolist()
    for i in range(max_t): config[sorted_indices[i]] = 'T'
    for i in range(max_t, max_t + max_d): config[sorted_indices[i]] = 'D'
        
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
