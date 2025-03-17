# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import pandas as pd
import argparse
import re

def extract_suffixes(df, base_columns):
    suffixes = set()
    pattern = re.compile(rf"^({'|'.join(map(re.escape, base_columns))})_(\w+)$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            suffixes.add(match.group(2))
    return list(suffixes)

def convert_csv_format(input_file, output_file, organs):
    # Define base input column names
    columns_join = {"kidney": ['left kidney', 'right kidney'],
                    "intestine": ['small bowel', 'large bowel']}
    
    # Read input CSV file
    df = pd.read_csv(input_file)
    
    # Determine suffixes in the dataset
    base_columns = [col for cols in columns_join.values() for col in cols]
    suffixes = extract_suffixes(df, base_columns)
    
    # Initialize an empty list for output rows
    outputs = []
    print(df.columns)
    
    for _, current_row in df.iterrows():
        organ_df = {'type_annotation': current_row['type_annotation']}
        
        for optional_col in ['labeler', 'subjectid_studyid', 'image1', 'study_id']:
            if optional_col in df.columns:
                organ_df[optional_col] = current_row[optional_col]
        
        for organ in organs:
            if organ in columns_join:
                for suffix in suffixes:
                    values_abnormal = [current_row[f"{col}_{suffix}"] for col in columns_join[organ]]
                    values_abnormal = [-1 if x!=x else x for x in values_abnormal]
                    x = max(values_abnormal)
                    organ_df[f"{organ}_{suffix}"] = None if x == -1 else x
            else:
                for suffix in suffixes:
                    column_name = f"{organ}_{suffix}"
                    if column_name in df.columns:
                        organ_df[column_name] = current_row[column_name]
                    elif organ in df.columns:
                        organ_df[organ] = current_row[organ]
        
        outputs.append(organ_df)
    
    # Convert to DataFrame and save
    output_df = pd.DataFrame(outputs)
    output_df.to_csv(output_file, index=False)
    print(f"File successfully converted and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV format for organ-specific columns with suffixes.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file")
    
    args = parser.parse_args()
    
    organs = ["spleen", "liver", "kidney", "stomach", "pancreas", "gallbladder", "intestine"]
    convert_csv_format(args.input_file, args.output_file, organs)