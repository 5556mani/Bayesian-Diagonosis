import pandas as pd
import numpy as np
from pathlib import Path

EPSILON = 1e-6
THRESHOLD = 0.90

def load_initial_info():
    current_dir = Path(__file__).resolve().parent
    data_path = current_dir.parent / "2_Extracting_features" / "Extracting_likelihood"
    priors = pd.read_csv(data_path / "disease_priors.csv", index_col=0)
    likelihoods = pd.read_csv(data_path / "likelihood_matrix.csv", index_col=0).replace(0, EPSILON)
    return priors['prior_probability'].copy(), likelihoods

def update_probabilities(current_probs, likelihood_matrix, pos_symptoms, neg_symptoms):
    updated_probs = current_probs.copy()
    for s in pos_symptoms:
        if s in likelihood_matrix.columns:
            updated_probs *= likelihood_matrix[s]
    for s in neg_symptoms:
        if s in likelihood_matrix.columns:
            updated_probs *= (1 - likelihood_matrix[s])
    if updated_probs.sum() > 0:
        updated_probs /= updated_probs.sum()
    return updated_probs

def find_next_symptom(current_probs, likelihood_matrix, asked_symptoms):
    top_10 = current_probs.sort_values(ascending=False).head(10)
    candidate_pool = set()
    for disease in top_10.index:
        top_s = likelihood_matrix.loc[disease].sort_values(ascending=False).head(10).index
        candidate_pool.update(top_s)
    
    # CRITICAL FIX: The logic now strictly ignores anything in asked_symptoms
    remaining = list(candidate_pool - asked_symptoms)
    if not remaining:
        return None
    s_scores = {s: (likelihood_matrix[s] * current_probs).sum() for s in remaining}
    return max(s_scores, key=s_scores.get)

def print_top_5(current_probs):
    print("\n--- Current Top 5 Disease Probabilities (Exact Values) ---")
    top_5 = current_probs.sort_values(ascending=False).head(5)
    for disease, prob in top_5.items():
        print(f"| {disease:<35} | {prob:.15f}")
    print("-" * 60)

def initiate_conversation():
    current_probs, likelihood_matrix = load_initial_info()
    asked_symptoms = set()
    
    print("\n" + "="*60)
    print("BAYESIAN MEDICAL DIAGNOSIS SYSTEM")
    print("="*60)

    # INITIAL INPUT
    print("\n[INSTRUCTION]: List all your initial symptoms (comma separated).")
    user_input = input("\n[Doctor]: Tell me, what symptoms are you currently facing?\n[User]: ")
    
    initial_positives = [s.strip() for s in user_input.split(',') if s.strip()]
    
    # Add initial symptoms to the 'asked' list immediately
    for s in initial_positives:
        asked_symptoms.add(s)
    
    current_probs = update_probabilities(current_probs, likelihood_matrix, initial_positives, [])
    print_top_5(current_probs)

    # ROUNDS 1+: FOLLOW-UP INQUIRY
    while current_probs.max() < THRESHOLD:
        next_q = find_next_symptom(current_probs, likelihood_matrix, asked_symptoms)
        
        if not next_q:
            print("\n[Doctor]: No more specific questions available. Finalizing diagnosis...")
            break
            
        print(f"\n[INSTRUCTION]: Answer 'yes' or 'no'. Add extra symptoms after a comma if needed.")
        response_raw = input(f"\n[Doctor]: Do you face {next_q}?\n[User]: ").lower().split(',')
        
        # Mark the question we just asked as discussed
        asked_symptoms.add(next_q)
        
        main_answer = response_raw[0].strip()
        extra_symptoms = [s.strip() for s in response_raw[1:] if s.strip()]
        
        pos_to_update = []
        neg_to_update = []

        # Process the direct Answer
        if main_answer == 'yes':
            pos_to_update.append(next_q)
        elif main_answer == 'no':
            neg_to_update.append(next_q)

        # Process extra symptoms and mark them as discussed too
        for s in extra_symptoms:
            pos_to_update.append(s)
            asked_symptoms.add(s)

        # Perform the update
        current_probs = update_probabilities(current_probs, likelihood_matrix, pos_to_update, neg_to_update)
        print_top_5(current_probs)

    final_disease = current_probs.idxmax()
    print("\n" + "*"*60)
    print(f"FINAL DIAGNOSIS: {final_disease}")
    print(f"EXACT CONFIDENCE: {current_probs.max():.15f}")
    print("*"*60)

if __name__ == "__main__":
    initiate_conversation()