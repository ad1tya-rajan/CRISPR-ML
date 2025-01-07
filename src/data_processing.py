# Data Processing

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

input_path = "data/raw/guideseq.csv"
output_path = "data/processed/processed_guideseq.csv"

def process_data(input_path, output_path):
    
    """
    Processes guide sequence data by performing data cleaning, validation, normalization, and encoding, 
    then saves the processed data to a CSV file.

    Parameters:
    input_path (str): The file path to the input CSV containing raw guide sequence data.
    output_path (str): The file path where the processed CSV data will be saved.

    """
    print("Loading data...")
    
    try:
        data = pd.read_csv(input_path)
        print(f"Dataset loaded successfully, with dim. {data.shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Cleaning data...")
    data = data.drop_duplicates()
    data = data.dropna()

    valid_bool_mask = data['On'].str.match(r'^[ACGTN]+$') & data['Off'].str.match(r'^[ACGTN]+$')        # check regex to validate nucleotide sequence
    data = data[valid_bool_mask]
    print("Data cleaned.")

    if 'Reads' in data.columns:
        print("Normalising 'Reads' column")
        scaler = MinMaxScaler()
        data['Reads'] = scaler.fit_transform(data['Reads'].values.reshape(-1, 1))

    if 'Active' in data.columns:
        print("One-hot encoding 'Active' column")
        data['Active'] = data['Active'].astype(int).clip(0, 1)

    try:
        data.to_csv(output_path, index = False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")