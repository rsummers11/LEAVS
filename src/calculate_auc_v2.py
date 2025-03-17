# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi
# Description:
# Auxiliary file called by create_raw_table.py.

import pandas as pd
import argparse
import os
import numpy as np
import re

def main(args):
    columns_to_use = args.abnormalities
    print(args.single_file, args.groundtruth)
    preds = pd.read_csv(args.single_file)
    if 'labeler' in preds.columns:
        preds = preds[preds['labeler'].astype(str)==args.labeler]
    annot = pd.read_csv(args.groundtruth)
    annot = annot[annot['type_annotation']==args.type_annotation]
    annot= annot.fillna(-2 if args.type_annotation=='labels' else -1)
    
    preds['subjectid_studyid'] = preds['subjectid_studyid'].astype(str).str.lstrip('_')
    if not 'image1' in annot.columns:
        annot['image1'] = annot['subjectid_studyid'].astype(str).apply(
            lambda x: x[:len(x)//2] if x[len(x)//2]=='_' else x.split('_')[-1]
        )
    annot['study_id'] = annot['image1'].astype(str).str.lstrip('_').str.rstrip('.txt').astype(str).str.rsplit('/', n=1).str[-1]

    preds = preds[preds['type_annotation']==args.type_annotation]
    preds= preds.fillna(-2 if args.type_annotation=='labels' else -1)

    preds['study_id'] = preds['subjectid_studyid'].astype(str).apply(
        lambda x: x[:len(x)//2] if x[len(x)//2]=='_' else x.split('_')[-1]
    ).astype(str).str.rstrip('.txt').str.rsplit('/', n=1).str[-1]
    print(annot['study_id'])
    print(preds['study_id'])

    annot = annot.sort_values(by='study_id')

    preds = pd.merge(annot['study_id'], preds, on='study_id', how='inner').reset_index(drop=True) 
    annot = annot[annot['study_id'].isin(preds['study_id'])].reset_index(drop=True)
    
    for column_name in annot.columns:
        if column_name in columns_to_use:
            annot[column_name] = pd.to_numeric(annot[column_name])
            preds[column_name] = pd.to_numeric(preds[column_name])

    dict_aucs = {}
    if args.type_annotation == 'labels':

        for column_name in annot.columns:
            print('oi1', column_name, column_name in columns_to_use)
            
            if column_name in columns_to_use:
                annot[column_name] = annot[column_name].replace(1, 4)
                annot[column_name] = annot[column_name].replace(-1, 3)
                annot[column_name] = annot[column_name].replace(-3, 2)
                annot[column_name] = annot[column_name].replace(-2, 1)

                preds[column_name] = preds[column_name].replace(1, 4)
                preds[column_name] = preds[column_name].replace(-1, 3)
                preds[column_name] = preds[column_name].replace(-3, 2)
                preds[column_name] = preds[column_name].replace(-2, 1)
                
                print(preds)
                print(column_name, len(preds[column_name].values))

                if 'labeler' in annot.columns and hasattr(args, "exclude_labeler"):
                    #only consider cases when the two other labelers agree on the presence or absence of fiding

                    # Function to map values to groups
                    def map_category(value):
                        if value in {4,3}:
                            return 'A'
                        elif value in {0,1,2}:
                            return 'B'

                    # Apply filtering
                    filtered_annot = annot.copy()
                    filtered_annot['original_index'] = filtered_annot.index
                    filtered_annot = filtered_annot[filtered_annot['labeler'] != args.exclude_labeler]
                    filtered_annot = filtered_annot.copy()

                    # Map column_to_use to categories
                    filtered_annot.loc[:, 'category'] = filtered_annot[column_name].map(map_category)

                    # Keep only study_id groups where all rows belong to the same category
                    valid_studies = (
                        filtered_annot.groupby('study_id')['category']
                        .transform(lambda x: x.nunique() == 1)
                    )

                    filtered_annot = filtered_annot[valid_studies]

                    if len(filtered_annot)>0:
                        # Drop the temporary category column if not needed
                        filtered_annot = filtered_annot.drop(columns=['category'])
                        
                        filtered_annot = filtered_annot.groupby('study_id', as_index=False)[[column_name, 'original_index']].first()
                        
                        valid_indices = filtered_annot['original_index']
                        filtered_preds = preds.loc[valid_indices]
                        assert(len(filtered_preds)==len(filtered_annot))
                        preds_to_use =  filtered_preds[column_name].values
                        annots_to_use = filtered_annot[column_name].values
                        ids_to_use = filtered_preds['study_id'].values
                    else:
                        preds_to_use =  np.array([])
                        annots_to_use = np.array([])
                        ids_to_use = np.array([])
                else:
                    preds_to_use =  preds[column_name].values
                    annots_to_use = annot[column_name].values
                    ids_to_use = preds['study_id'].values

                annots_to_use[annots_to_use==1] = 0
                annots_to_use[annots_to_use==3] = 1
                annots_to_use[annots_to_use==2] = 0
                annots_to_use[annots_to_use==4] = 1		

                preds_to_use[preds_to_use==1] = 0
                preds_to_use[preds_to_use==3] = 1
                preds_to_use[preds_to_use==2] = 0
                preds_to_use[preds_to_use==4] = 1
                
                dict_aucs[column_name + '_annotpred'] = (ids_to_use, annots_to_use, preds_to_use)
        return dict_aucs
    elif args.type_annotation == 'urgency':
        for column_name in annot.columns:
            if column_name in columns_to_use:
                if 'labeler' in annot.columns and hasattr(args, "exclude_labeler"):
                    # Apply filtering
                    filtered_annot = annot.copy()
                    filtered_annot['original_index'] = filtered_annot.index
                    filtered_annot = filtered_annot.copy()

                    # Keep only study_id groups where all rows belong to the same category
                    valid_studies = (
                        filtered_annot.groupby('study_id')[column_name]
                        .transform(lambda x: (x >= 0).sum() >= 3)
                    )

                    filtered_annot = filtered_annot[valid_studies]

                    filtered_annot = filtered_annot[filtered_annot['labeler'] != args.exclude_labeler]
                    filtered_annot = filtered_annot.copy()

                    filtered_annot = filtered_annot.groupby('study_id', as_index=False).agg(
                            {column_name: 'mean', 'original_index': 'first'}
                        )
                    
                    valid_indices = filtered_annot['original_index']
                    filtered_preds = preds.loc[valid_indices]
                    assert(len(filtered_preds)==len(filtered_annot))
                    preds_to_use =  filtered_preds[column_name].values
                    annots_to_use = filtered_annot[column_name].values
                    ids_to_use = filtered_preds['study_id'].values
                else:
                    preds_to_use =  preds[column_name].values
                    annots_to_use = annot[column_name].values
                    ids_to_use = preds['study_id'].values
                dict_aucs[column_name + '_annotpred'] = (ids_to_use, annots_to_use, preds_to_use)
        return dict_aucs
        

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_annotation", type=str, choices = ["probability", "labels"], default="labels", help='''''')
    parser.add_argument("--run_labels", type=str2bool, default='true', help='''''')
    parser.add_argument('--single_file', type=str, default = './parsing_results_llm.csv', help='''Path to file to run the script for. For example, 
    use "--single_file=./parsing_results.csv" to run the script for the ./parsing_results.csv file.
    . Default: ./parsing_results_llm.csv''')
    parser.add_argument('--groundtruth', type=str, default = './human_annotation.csv', help='''Path to file containing the groundtruth for all 
    reports that you are testing for. The ID (filename) of a report should be in the image2 column. The abnormality labels should be in columns named
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion", "Lung Opacity",  
    "Pleural Effusion", "Pleural Other", "Pneumonia", "Pneumothorax", "Support Devices". Default: ./human_annotation.csv''')
    parser.add_argument("--labeler", type=str, choices = ["chexpert", "vqa", 'llm', 'vicuna', 'llm_generic'], default=None, help='''''')
    parser.add_argument("--dataset", type=str, choices = ['nih', 'mimic'], default=None, help='''''')
    parser.add_argument("--abnormalities", type=str, default=None, nargs = '+', help='''''')

    args = parser.parse_args()
    main(args)