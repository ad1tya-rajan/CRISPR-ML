import pandas as pd
import numpy as np

from data_processing import process_data

#TODO: levenshtein distance works in feature_eng, but not in eda - fix 

def levenshtein_distance(seq1, seq2):

    """
    Computes the Levenshtein distance between two sequences.

    Parameters
    ----------
    seq1 (str): The first sequence.
    seq2 (str): The second sequence.

    Returns
    -------
    int: The Levenshtein distance between the two sequences.

    """

    len1 = len(seq1)
    len2 = len(seq2)

    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)      # 2D array for dynamic programming

    for i in range(len1 + 1):
        dp[i][0] = i

    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1

    return dp[len1][len2]

def get_lev_distance(row):
    return levenshtein_distance(row['On'], row['Off'])

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
    data['Levenshtein_Distance'] = data.apply(get_lev_distance, axis = 1)

    data.to_csv(output_path, index = False)
    print(f"Features saved successfully to {output_path}")

if __name__ == "__main__":
    input_path = r'C:\Users\adity\Projects\CRISPR-ML\data\raw\guideseq.csv'
    output_path = r'C:\Users\adity\Projects\CRISPR-ML\data\processed\features_guideseq.csv'
    extract_features(input_path, output_path)
    