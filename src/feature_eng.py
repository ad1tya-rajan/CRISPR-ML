import pandas as pd
import numpy as np

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
    
    data = pd.read_csv(input_path)

    data['Guide_Length'] = data['On'].apply(len)
    data['On_GC_Content'] = data['On'].apply(get_gc_content)
    data['Off_GC_Content'] = data['Off'].apply(get_gc_content)

    data.to_csv(output_path, index = False)
    print(f"Features saved successfully to {output_path}")

if __name__ == "__main__":
    input_path = 'data/raw/guideseq.csv'
    output_path = 'data/processed/guideseq_features.csv'
    extract_features(input_path, output_path)
    