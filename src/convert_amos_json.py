# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import json
import pandas as pd
import os

organs_dict = {"spleen":"spleen", "liver":"liver", "right kidney":"right kidney", "left kidney":"left kidney", \
            "stomach":"stomach", "pancreas":"pancreas", "gallblader":"gallbladder", "small bowel":"small bowel", "large bowel":"large bowel"}
organs = ["spleen", "liver", "right kidney", "left kidney", \
            "stomach", "pancreas", "gallblader", "small bowel", "large bowel"]
finding_types = ["quality", "postsurgical_absent", "enlarged_atrophy", "diffuse", "focal"]
urgency = {"":"", "normal":0, "low urgency":1, "medium urgency":2, "high urgency":3}

records = []
for index_labeler, labeler in enumerate(['1', '2', '3', '4', '5']):
    for folder_name in [str(index_folder) for index_folder in range(1,21)]:
        if not os.path.exists(f'amos_annotations/{labeler}/{folder_name}/results.json'):
            continue
        with open(f'amos_annotations/{labeler}/{folder_name}/results.json', 'r') as jsonfile:
            data = json.load(jsonfile)
        
        for filename, details in data.items():
            filename_other = os.path.basename(filename)
            if '\\' in filename_other:
                filename_other = filename_other.split('\\')[1]
            record = {'image1': filename_other, 'subjectid_studyid': filename_other+'_'+filename_other, 'type_annotation':'labels', 'labeler': index_labeler}
            urgency_row = {'image1': filename_other, 'subjectid_studyid': filename_other+'_'+filename_other, 'type_annotation':'urgency', 'labeler': index_labeler}
            for organ_index, organ_labels in enumerate(details['label_frames']):
                for finding_type_index, finding_label in enumerate(organ_labels['checkboxes']):
                    record[f'{organs_dict[organs[organ_index]]}_{finding_types[finding_type_index]}'] = finding_label*1
                for urgency_index, urgency_label in enumerate(["",""] + organ_labels['dropdowns']):
                    if record[f'{organs_dict[organs[organ_index]]}_{finding_types[urgency_index]}']:
                        urgency_row[f'{organs_dict[organs[organ_index]]}_{finding_types[urgency_index]}'] = urgency[urgency_label]
            # record['note'] = details['note']
            records.append(record)
            records.append(urgency_row)

# Create DataFrame
df = pd.DataFrame(records)

# Write DataFrame to CSV
df.to_csv('amos_test_annotations.csv', index=False)