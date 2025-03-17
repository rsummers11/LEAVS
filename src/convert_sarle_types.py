# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import pickle
# from global_ import abnormality_denominations
import pandas as pd
import csv

abnormality_denominations = ["spleen",
                              "liver",
                              "right kidney",
                              "left kidney",
                              "stomach",
                              "pancreas",
                              "gallbladder",
                              "small bowel",
                              "large bowel",]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

csv_dict = {}
with open('sarle_convert_labels_dict.csv', 'r') as file:
    reader = csv.reader(file)
    csv_dict = {row[0]: row[1] for row in reader}
finding_types_new_names = {'diffusefocalenlarged':'enlarged_atrophy_diffuse_focal', 'device':'device', 'postsurgical':'postsurgical_absent', 'other':'other', 'anatomy':'anatomy'}

all_finding_types = set(csv_dict.values())
all_finding_types.remove('na')
locations = pickle.load(open('sarle_predictions_amos.pkl','rb'))
final_df = []
for report_file in locations:
    location = locations[report_file]
    row_dict = {'image1': report_file, 'type_annotation': 'labels'}
    for organ in abnormality_denominations:
        if organ == 'right kidney':
            organ = 'kidney'
        if organ =='left kidney':
            continue
        if organ == 'small bowel':
            organ = 'intestine'
        if organ =='large bowel':
            continue
        findings_found = (location[organ][location[organ] != 0].index.tolist())
        converted_findings = set([csv_dict[finding_found] for finding_found in findings_found if csv_dict[finding_found] !='na'])
        row_dict.update({f'{organ}_{finding_types_new_names[finding_type]}': 1 if finding_type in converted_findings else 0 for finding_type in all_finding_types})
    final_df.append(row_dict)
pd.DataFrame(final_df).to_csv('converted_sarle_labels_testamos.csv', index = False)