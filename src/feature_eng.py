import pandas as pd
import numpy as np

from data_processing import process_data

def get_gc_content(sequence):
    
    """
    Calculates the GC content of a given DNA sequence.

    Parameters:
    sequence (str): A string representing a DNA sequence

    Returns:
    gc_content (float): The GC content of the sequence in percent

    """

    gc_count = sequence.count('G') + sequence.count('C')
    gc_content = (gc_count / len(sequence)) * 100
    return gc_content

def extract_features(input_path, output_path):

    """
    Extracts features from guide sequence data and saves to CSV.

    Parameters:
    input_path (str): The file path to the input CSV containing guide sequences.
    output_path (str): The file path where the output CSV with features will be saved.

    """
    processed_path = r'C:\Users\adity\Projects\CRISPR-ML\data\processed\processed_guideseq.csv'
    process_data(input_path, processed_path)
    data = pd.read_csv(processed_path)

    data['Guide_Length'] = data['On'].apply(len)
    data['On_GC_Content'] = data['On'].apply(get_gc_content)
    data['Off_GC_Content'] = data['Off'].apply(get_gc_content)

    data.to_csv(output_path, index = False)
    print(f"Features saved successfully to {output_path}")

if __name__ == "__main__":
    input_path = r'C:\Users\adity\Projects\CRISPR-ML\data\raw\guideseq.csv'
    output_path = r'C:\Users\adity\Projects\CRISPR-ML\data\processed\features_guideseq.csv'
    extract_features(input_path, output_path)
    