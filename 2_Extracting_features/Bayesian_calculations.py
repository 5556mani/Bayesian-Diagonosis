import pandas as pd
import os

def prepare_and_save_bayesian_components(file_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # Load the population data
    df = pd.read_csv(file_path)
    total_population = len(df)
    
    # 1. THE PRIOR FOR ALL DISEASES
    prior_for_all_diseases = df['diseases'].value_counts() / total_population
    prior_for_all_diseases.to_csv(os.path.join(output_folder, "disease_priors.csv"), header=['prior_probability'])
    
    # 2. THE UPDATE MATRIX (LIKELIHOODS)
    update_matrix = df.groupby('diseases').mean()
    update_matrix.to_csv(os.path.join(output_folder, "likelihood_matrix.csv"))
    
    # 3. THE PROBABILITY OF THE SYMPTOMS (EVIDENCE)
    symptoms_only = df.drop(columns=['diseases'])
    probability_of_symptoms = symptoms_only.sum() / total_population
    probability_of_symptoms.to_csv(os.path.join(output_folder, "symptom_probabilities.csv"), header=['symptom_probability'])

    print(f"Success! All 3 Bayesian components saved in '{output_folder}'")

# Dynamic Path Logic 

# 1. Get the directory where THIS script is currently sitting
# This points to '.../Extracting_features'
current_script_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Define the output directory INSIDE this current folder
output_dir = os.path.join(current_script_dir, "Extracting_likelihood")

# 3. Locate the input data by going UP one level to the project root
project_root = os.path.dirname(current_script_dir)
input_csv = os.path.join(project_root, "1_Installing_Data", "data", "Diseases_and_Symptoms_dataset.csv")

# Execution 
if os.path.exists(input_csv):
    prepare_and_save_bayesian_components(input_csv, output_dir)
else:
    print(f"Error: Could not find raw data at: {input_csv}")