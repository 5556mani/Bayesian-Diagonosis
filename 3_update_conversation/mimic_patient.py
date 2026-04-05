import pandas as pd
from pathlib import Path
import random
import mimic_doctor as engine 

def run_mimic_simulation(num_runs):
    # DATA LOADING
    current_dir = Path(__file__).resolve().parent
    # Path to the raw CSV containing the ground truth rows
    raw_data_path = current_dir.parent / "1_Installing_Data" / "data" / "Diseases_and_Symptoms_dataset.csv"
    
    if not raw_data_path.exists():
        print(f"Error: Could not find raw data at {raw_data_path}")
        return

    raw_df = pd.read_csv(raw_data_path)
    symptom_columns = [col for col in raw_df.columns if col != 'diseases']
    
    results_log = []

    print(f"\n{'='*60}")
    print(f"STARTING MIMIC SIMULATION: {num_runs} PATIENTS")
    print(f"{'='*60}")

    for i in range(num_runs):
        # 1. Select a random patient truth from the dataset
        patient_row = raw_df.sample(n=1).iloc[0]
        actual_disease = str(patient_row['diseases']).strip()
        
        # 2. Reset the Doctor's state for this specific patient
        current_probs, likelihood_matrix = engine.load_initial_info()
        asked_symptoms = set()
        
        print(f"\n\n>>> PATIENT {i+1} (Truth: {actual_disease})")
        print("-" * 40)

        # 3. Identify the FIRST symptom for the initial question
        first_symptom = None
        for s in symptom_columns:
            if patient_row[s] == 1:
                first_symptom = s
                break
        
        if not first_symptom:
            print("[System]: Patient has no symptoms recorded. Skipping.")
            continue

        # THE DIALOGUE FLOW
        # Initial Round
        print(f"[Doctor]: Tell me, what symptoms are you currently facing?")
        print(f"[Mimic Patient]: {first_symptom}")
        
        asked_symptoms.add(first_symptom)
        current_probs = engine.update_probabilities(current_probs, likelihood_matrix, [first_symptom], [])
        engine.print_top_5(current_probs)

        rounds = 1
        # Follow-up inquiries until THRESHOLD (0.90) is reached
        while current_probs.max() < engine.THRESHOLD:
            next_q = engine.find_next_symptom(current_probs, likelihood_matrix, asked_symptoms)
            
            if not next_q:
                print("\n[Doctor]: I have no further specific questions.")
                break
            
            asked_symptoms.add(next_q)
            rounds += 1
            
            print(f"\n[Doctor]: Do you face {next_q}?")
            
            # MIMIC LOGIC: Look up the truth in the selected row
            if patient_row[next_q] == 1:
                print(f"[Mimic Patient]: yes")
                current_probs = engine.update_probabilities(current_probs, likelihood_matrix, [next_q], [])
            else:
                print(f"[Mimic Patient]: no")
                current_probs = engine.update_probabilities(current_probs, likelihood_matrix, [], [next_q])
            
            engine.print_top_5(current_probs)

        # 4. Record Results for this patient
        predicted_disease = str(current_probs.idxmax()).strip()
        confidence = current_probs.max()
        is_correct = (predicted_disease.lower() == actual_disease.lower())

        results_log.append({
            "Actual": actual_disease,
            "Predicted": predicted_disease,
            "Rounds": rounds,
            "Correct": is_correct
        })

        print("\n" + "*"*40)
        print(f"DIAGNOSIS COMPLETE FOR PATIENT {i+1}")
        print(f"Actual: {actual_disease} | Predicted: {predicted_disease}")
        print(f"Exact Confidence: {confidence:.15f}")
        print("*"*40)

    # FINAL AGGREGATE SUMMARY 
    summary_df = pd.DataFrame(results_log)
    accuracy = summary_df['Correct'].mean() * 100
    avg_rounds = summary_df['Rounds'].mean()

    print("\n" + "="*60)
    print(f"FINAL SIMULATION SUMMARY ({num_runs} RUNS)")
    print("="*60)
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Average Inquiry Rounds: {avg_rounds:.2f}")
    print("-" * 60)
    # Display the comparison list as requested
    print(summary_df[['Actual', 'Predicted', 'Correct']])
    print("="*60)

if __name__ == "__main__":
    n_input = input("Enter number of patients to mimic: ")
    run_mimic_simulation(int(n_input))