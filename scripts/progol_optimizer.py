import numpy as np
import pandas as pd

def calculate_ticket_cost(doubles, triples):
    return (2**doubles * 3**triples) * 15

def get_entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-9))

def get_custom_ticket_config(match_probs, num_doubles, num_triples):
    match_stats = [{'id': i, 'entropy': get_entropy(p)} for i, p in enumerate(match_probs)]
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    sorted_indices = df.index.tolist()
    config = ['S'] * len(match_probs)
    for i in range(min(num_triples, len(sorted_indices))): config[sorted_indices[i]] = 'T'
    for i in range(num_triples, min(num_triples + num_doubles, len(sorted_indices))): config[sorted_indices[i]] = 'D'
    return config, calculate_ticket_cost(num_doubles, num_triples)

def optimize_progol_ticket(match_probs, budget=2000):
    match_stats = [{'id': i, 'entropy': get_entropy(p)} for i, p in enumerate(match_probs)]
    df = pd.DataFrame(match_stats).sort_values('entropy', ascending=False)
    max_d, max_t = 0, 0
    for t in range(10):
        for d in range(14): 
            cost = calculate_ticket_cost(d, t)
            if cost <= budget and (d + t) <= len(match_probs):
                if cost > calculate_ticket_cost(max_d, max_t): max_d, max_t = d, t
    config = ['S'] * len(match_probs)
    sorted_indices = df.index.tolist()
    for i in range(max_t): config[sorted_indices[i]] = 'T'
    for i in range(max_t, max_t + max_d): config[sorted_indices[i]] = 'D'
    return config, calculate_ticket_cost(max_d, max_t), max_d, max_t

def print_final_ticket(match_ids, probs, config):
    print("\n" + "="*95)
    print(f"{'GAME':<5} | {'MATCHUP':<40} | {'MARK THIS':<10} | {'TYPE':<5} | {'WIN PROB'}")
    print("-" * 95)
    
    for i, (mid, p, c) in enumerate(zip(match_ids, probs, config)):
        # Labels: 0=L, 1=E, 2=V
        sorted_idxs = np.argsort(p)[::-1] # indices of highest probs
        
        if c == 'S':
            mark = {0:'L', 1:'E', 2:'V'}[sorted_idxs[0]]
            prob_cont = p[sorted_idxs[0]]
        elif c == 'D':
            # Take the top 2 highest probability outcomes
            marks = sorted([sorted_idxs[0], sorted_idxs[1]])
            mark = "/".join([{0:'L', 1:'E', 2:'V'}[x] for x in marks])
            prob_cont = p[sorted_idxs[0]] + p[sorted_idxs[1]]
        else:
            mark = "L/E/V"
            prob_cont = 1.0
            
        print(f"{i+1:2}    | {str(mid):<40} | {mark:<10} | {c:<5} | {prob_cont*100:6.1f}%")
    print("="*95)
