# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import pandas as pd
import argparse
import numpy as np

def main(input_file):
    # Load CSV files into pandas DataFrames
    df = pd.read_csv(input_file)

    df.fillna(float('inf'))

    # Function to compute majority vote for each cell
    def majority_vote(series):
        return (series.sum() >= (len(series) / 2.)).astype(int)
    
    # Function to compute majority vote for each cell
    def vote_urgency(series):
        # Filter dataframes based on type_annotation
        labels_filtered = series[series['type_annotation'] == 'labels'].sort_values('labeler')
        urgency_filtered = series[series['type_annotation'] == 'urgency'].sort_values('labeler')
        
        # Identify numerical columns to apply voting
        relevant_columns = [col for col in series.columns if col not in ['subjectid_studyid', 'type_annotation', 'image1', 'labeler']]
        
        # Extract values for computation
        labels_values = labels_filtered[relevant_columns].values
        urgency_values = urgency_filtered[relevant_columns].values
        
        # Ensure arrays are aligned and compute weighted sum
        # result_values = np.sum(labels_values * urgency_values, axis=0) / (np.sum(labels_values, axis=0)*(np.sum(labels_values, axis=0) >= (labels_values.shape[0] / 2.)))
        print(urgency_values, labels_values)
        result_values = np.sum(labels_values * urgency_values, axis=0) / (np.sum(labels_values, axis=0)*(np.sum(labels_values, axis=0) == (labels_values.shape[0])))


        # Create a result DataFrame that retains all columns
        result_df = pd.DataFrame(columns=series.columns, index=[series.index[0]])  # Keep index format
        
        # Fill in computed values
        result_df[relevant_columns] = result_values
        # result_df[['image1']] = series[['image1']].iloc[0]  # Keep metadata
        result_df[['type_annotation']] = 'urgency'  # Keep metadata

        return result_df.iloc[0]
    # print(combined_df)

    # Apply majority vote function to each cell excluding the id column
    if not 'image1' in df.columns:
        df['image1'] = df['subjectid_studyid'].astype(str).apply(
            lambda x: x[:len(x)//2] if x[len(x)//2]=='_' else x.split('_')[-1]
        )
    majority_vote_df = df[df['type_annotation']=='labels'].groupby(['image1','subjectid_studyid', 'type_annotation']).apply(lambda x: x.apply(majority_vote, axis=0), include_groups = True).reset_index()
    vote_urgency_df = df.fillna(0).groupby('image1').apply(vote_urgency, include_groups = False).reset_index()
    # urgency_df = df[df['type_annotation']=='labels'].groupby('filename').apply(lambda x: x.apply(majority_vote, axis=0))

    combined_df = pd.concat([majority_vote_df, vote_urgency_df], axis=0, ignore_index=True)
    combined_df.replace([np.inf, -np.inf], "", inplace=True)
    # Reset index to bring id back as a column
    combined_df.reset_index(inplace=True)
    combined_df = combined_df.drop('labeler', axis=1) 
    # Write the resulting DataFrame to a CSV file
    return combined_df
    

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Convert CSV format for organ-specific columns.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file")
    args = parser.parse_args()

    majority_vote_df = main(args.input_file)
    majority_vote_df.to_csv(args.output_file, index=False)