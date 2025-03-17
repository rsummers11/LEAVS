# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import pandas as pd
import argparse

def convert_csv_format(input_file, output_file, organs):
    # Define base input column names

    columns_join = {"kidney":['left kidney', 'right kidney'], 'intestine':['small bowel', 'large bowel']}

    # Read input CSV file
    df = pd.read_csv(input_file)
    
    # Initialize an empty dataframe with output columns
    outputs = []
    print(df.columns)
    for _,current_row in df.iterrows():
        # Process each organ and pivot the data
        organ_df = {'type_annotation':current_row['type_annotation']}
        if 'labeler' in df.columns:
            organ_df['labeler'] = current_row['labeler']
        if 'subjectid_studyid' in df.columns:
            organ_df['subjectid_studyid'] = current_row['subjectid_studyid']
        if 'image1' in df.columns:
            organ_df['image1'] = current_row['image1']
        if 'study_id' in df.columns:
            organ_df['study_id'] = current_row['study_id']
        for organ in organs:
            if organ in columns_join:
                
                values_abnormal = [current_row[column_consider] for column_consider in columns_join[organ]]
                values_abnormal = [-1 if x!=x else x for x in values_abnormal]
                x = max(values_abnormal)
                None if x==-1 else x
                organ_df[organ] = x
            else:
                organ_df[organ] = current_row[organ]

        
        # Merge into the output dataframe based on common keys
        outputs.append(organ_df)

    # Ensure all output columns exist
    output_df = pd.DataFrame(outputs)
    
    # Write to output CSV file
    output_df.to_csv(output_file, index=False)
    print(f"File successfully converted and saved to {output_file}")

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Convert CSV format for organ-specific columns.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file")

    args = parser.parse_args()

    organs = ["spleen",
                              "liver",
                              "kidney",
                              "stomach",
                              "pancreas",
                              "gallbladder",
                              "intestine"]
    convert_csv_format(args.input_file, args.output_file, organs)