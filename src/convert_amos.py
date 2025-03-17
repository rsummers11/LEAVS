# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import json
import pandas as pd

# Open the JSON file
with open('amos_dataset.json', 'r') as file:
    data = json.load(file)
dics_values = []
for split in ["training", "validation", "testing"]:
    for element in data[split]:
        id = element["image"]
        if "labels" in element.keys():
            dic_report = element["labels"]["report"]
            report = ""
            for key_l1 in dic_report.keys():
                report +=  key_l1 + ":\n"
                for key_l2 in dic_report[key_l1].keys():
                    report_part = dic_report[key_l1][key_l2]
                    if isinstance(report_part, list):
                        report_part = ' '.join(report_part)
                    if len(report_part)>0:
                        report += "\n" + key_l2 + ":\n"
                        report += report_part
                report += "\n\n"
            dics_values.append({"image1": id, "image2": id,"split": split, "anonymized_report": report})


# Create DataFrame
df = pd.DataFrame(dics_values)

# Save DataFrame to a CSV file
df.to_csv('amos_dataset_converted.csv', index=False)