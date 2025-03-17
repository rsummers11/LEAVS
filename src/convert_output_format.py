# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import pandas as pd
import argparse

def convert_csv_format(input_file, organs):
    # Define base input column names
    base_columns = [
        "subjectid_studyid", "organ", "report", "type_annotation"
    ]
    organ_columns = ["normal", "absent", "adjacent", "quality", "postsurgical",
                     "anatomy", "device", "enlarged", "atrophy", "diffuse", "focal"]
    
    base_columns = base_columns + organ_columns
    # Generate dynamic output columns based on organs
    output_columns = ["subjectid_studyid", "report", "type_annotation"]
    for organ in organs:
        output_columns.extend([f"{organ}_{col}" for col in organ_columns])
    
    # Read input CSV file
    print(base_columns)
    df = pd.read_csv(input_file, usecols=base_columns)
    # Initialize an empty dataframe with output columns
    output_df = pd.DataFrame(columns=output_columns)
    
    # Process each organ and pivot the data
    for organ in organs:
        organ_df = df[df['organ'] == organ]
        organ_prefix = f"{organ}_"
        
        # Rename columns to include the organ prefix
        renamed_columns = {col: organ_prefix + col for col in organ_columns}
        organ_df = organ_df.rename(columns=renamed_columns)
        
        # Drop the 'organ' column
        organ_df = organ_df.drop(columns=['organ'])
        
        # Merge into the output dataframe based on common keys
        if output_df.empty:
            output_df = organ_df
        else:

            output_df = pd.merge(
                output_df, organ_df, 
                on=["subjectid_studyid", "report", "type_annotation"], 
                how="outer"
            )
    
    # Ensure all output columns exist
    output_df = output_df.reindex(columns=output_columns)
    output_df.columns = [col.replace("_Unnamed: 4", "") for col in output_df.columns]
    return output_df

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Convert CSV format for organ-specific columns.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file")

    args = parser.parse_args()

    from global_ import organ_denominations as organs
    output_df = convert_csv_format(args.input_file, organs)
    # Write to output CSV file
    output_df.to_csv(args.output_file, index=False)