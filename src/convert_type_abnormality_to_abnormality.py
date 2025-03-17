# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import pandas as pd
import argparse

def convert_list_to_ordered_labels(list_numbers):
    # Define the mapping
    mapping = {1: 4, -1: 3, -3: 2, -2: 1, 0:0}
    # Use list comprehension for concise mapping
    return [mapping.get(annot, -1) for annot in list_numbers]
    
def convert_list_to_unordered_labels(list_numbers):
    # Define the mapping
    mapping = {4:1, 3: -1, 2: -3, 1: -2, 0:0, -1: None}
    # Use list comprehension for concise mapping
    return mapping[list_numbers]

def convert_list_to_ordered_urgency(list_numbers):
    # Define the mapping
    return [-1 if x!=x else x for x in list_numbers]

def convert_list_to_unordered_urgency(x):
    # Define the mapping
    return None if x==-1 else x

def convert_csv_format(df, organs, dataset, set_sarle = ''):
    if len(set_sarle)>0:
        if set_sarle=='sarle':
            if dataset=='amos':
                columns_consider = {"abnormality": ["enlarged_atrophy_diffuse_focal","postsurgical_absent","postsurgical","absent", "enlarged", "atrophy", "enlarged_atrophy", "diffuse", "focal"], 
                "postsurgical_absent":["postsurgical_absent","postsurgical","absent"],
                "enlarged_atrophy_diffuse_focal":["enlarged", "atrophy", "enlarged_atrophy", "enlarged_atrophy_diffuse_focal", "diffuse", "focal"]}
            elif dataset=='deep':
                columns_consider = {"abnormality": ["anatomy","device","postsurgical_absent", "postsurgical","absent", "enlarged", "atrophy", "diffuse", "focal", "enlarged_atrophy_diffuse_focal"]}
        elif set_sarle=='other':
            if dataset=='amos':
                columns_consider = {"abnormality": ["enlarged_atrophy_diffuse_focal","postsurgical_absent","postsurgical","absent", "enlarged", "atrophy", "enlarged_atrophy", "diffuse", "focal", "other"], 
                "postsurgical_absent":["postsurgical_absent","postsurgical","absent"],
                "enlarged_atrophy_diffuse_focal":["enlarged", "atrophy", "enlarged_atrophy", "enlarged_atrophy_diffuse_focal", "diffuse", "focal"]}
            elif dataset=='deep':
                columns_consider = {"abnormality": ["anatomy","device","postsurgical_absent", "postsurgical","absent", "enlarged", "atrophy", "diffuse", "focal", "enlarged_atrophy_diffuse_focal", "other"]}
    else:
        if dataset=='amos':
            columns_consider = {"abnormality": ["postsurgical_absent","postsurgical","absent", "enlarged", "atrophy", "enlarged_atrophy", "diffuse", "focal"], 
                "postsurgical_absent":["postsurgical_absent","postsurgical","absent"],
                "enlarged_atrophy":["enlarged", "atrophy", "enlarged_atrophy"],
                "diffuse":["diffuse"],
                "focal":["focal"]}
        elif dataset=='deep':
            columns_consider = {"abnormality": ["anatomy","device","postsurgical","absent", "enlarged", "atrophy", "diffuse", "focal"], 
                "postsurgical_absent":["postsurgical","absent"],
                "enlarged_atrophy":["enlarged", "atrophy", "anatomy"],
                "diffuse":["diffuse"],
                "focal":["focal"],
                'device': ['device']}
        
    # Initialize an empty dataframe with output columns
    outputs = []
    if 'subjectid_studyid' in df.columns:
        id_column_name = 'subjectid_studyid'
    elif 'image1' in df.columns:
        id_column_name = 'image1'
    for _,current_row in df.iterrows():
        # Process each organ and pivot the data
        organ_df = {id_column_name:current_row[id_column_name], 
                    'type_annotation':current_row['type_annotation']}
        if 'labeler' in df.columns:
            organ_df['labeler'] = current_row['labeler']
        for organ in organs:
            for finding_type in columns_consider:
                values_abnormal = [current_row[f'{organ}_{column_consider}'] for column_consider in columns_consider[finding_type] if f'{organ}_{column_consider}' in df.columns]
                if current_row['type_annotation']=='labels':
                    values_abnormal = convert_list_to_ordered_labels(values_abnormal)
                else:
                    values_abnormal = convert_list_to_ordered_urgency(values_abnormal)
                values_abnormal = max(values_abnormal)
                if current_row['type_annotation']=='labels':
                    values_abnormal = convert_list_to_unordered_labels(values_abnormal)
                else:
                    values_abnormal = convert_list_to_unordered_urgency(values_abnormal)
                if finding_type=='abnormality':
                    organ_df[f'{organ}'] = values_abnormal
                else:
                    organ_df[f'{organ}_{finding_type}'] = values_abnormal

        # Merge into the output dataframe based on common keys
        outputs.append(organ_df)

    # Ensure all output columns exist
    output_df = pd.DataFrame(outputs)
    return output_df

if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Convert CSV format for organ-specific columns.")
    parser.add_argument("--input_file", type=str, help="Path to the input CSV file")
    parser.add_argument("--output_file", type=str, help="Path to the output CSV file")
    parser.add_argument("--dataset", type=str, default = 'deep')
    args = parser.parse_args()

    organs = ["spleen",
                "liver",
                "right kidney",
                "left kidney",
                "stomach",
                "pancreas",
                "gallbladder",
                "small bowel",
                "large bowel",]

    # Read input CSV file
    df = pd.read_csv(args.input_file)
    

    output_df = convert_csv_format(df, organs, args.dataset)

    # Write to output CSV file
    output_df.to_csv(args.output_file, index=False)
    print(f"File successfully converted and saved to {args.output_file}")