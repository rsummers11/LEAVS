# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

# Use `python <python_file> --help` to check inputs and outputs
# Description:
# Files that converts outputs from labelers and classifiers into
# scores (precision, recall, f1, mae or auc) with confidence intervals
# and p-values comparing the scores for the proposed labeler/classifier 
# LEAVS against the baselines using bootstrapping

from calculate_auc_v2 import main as get_labels
from types import SimpleNamespace
import pandas as pd
import numpy as np
from collections import defaultdict
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score,f1_score, matthews_corrcoef
from scipy.stats import kendalltau, mode
from sklearn.metrics import confusion_matrix
from pathlib import Path
import random
import copy

n_iterations = 2000

def specificity_score(target_column, output_column):
    tn, fp, fn, tp = confusion_matrix(target_column, output_column).ravel()
    return tn / (tn+fp+1e-20)

def kendalltau_metric(target_column, output_column):
    if len(output_column)==0:
        return 0
    assert((target_column>=0).all())
    assert((output_column>=0).all())
    result = kendalltau(target_column, output_column).statistic
    if result!=result:
        result = 0
    return result

def auc_metric(target_column, output_column):
    if len(target_column)==0:
        return float('inf')
    if target_column.std()==0:
        return float('inf')
    return roc_auc_score(target_column, output_column, average=None)
    
def find_full_folder_name(base_folder, partial_name):
    # List all entries in the base folder
    for entry in os.listdir(base_folder):
        # Construct the full path
        full_path = os.path.join(base_folder, entry)
        # Check if this entry is a directory and if the partial name is in this entry's name
        if os.path.isdir(full_path) and partial_name in entry:
            return entry
    return None

def get_if_several_predictions(vector):
    return isinstance(vector,list) or len(vector.shape)==2

def get_macro_bootstrap(vector_sum, count_n):
    vector_avg = vector_sum/count_n
    sorted_scores = np.array(vector_avg)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    median = sorted_scores[int(0.5 * len(sorted_scores))]
    average = sorted_scores.mean()
    var = sorted_scores.var()

    return median, confidence_lower, confidence_upper, average, var, np.array(vector_avg)

def get_hypothesis_test_equality_macro(vector_sum1, vector_sum2, count_n, median_1, median_2):
    vector_avg1 = vector_sum1/count_n
    vector_avg2 = vector_sum2/count_n
    return get_hypothesis_test_equality(vector_avg1, vector_avg2, median_1, median_2)

def get_hypothesis_test_equality(bootstrapped_scores1, bootstrapped_scores2, median_1, median_2):
    count_p = 0.
    for value in (bootstrapped_scores1 - bootstrapped_scores2):
        if value>=0:
            if value==0:
                count_p+= 0.5
            else:
                count_p+= 1
        else:
            count_p+= 0
    p_value = count_p/len(bootstrapped_scores2)
    if p_value>0.5:
        p_value = 1 - p_value
    p_value = 2*p_value
    return p_value

from joblib import Parallel, delayed
import joblib

import hashlib
def get_hashed_seed(global_seed, index):
    hash_input = f"{global_seed}_{index}".encode()
    hash_output = hashlib.sha256(hash_input).hexdigest()
    # Convert the hexadecimal hash to an integer
    return int(hash_output, 16) % (2**32 - 1)

def bootstrap_iteration(gt_vector_list, pred_vector_list, scores_fn, weights_list, n_samples, gt_vector_length, seed, index_process):
    weights = weights_list[0]
    gt_vector = gt_vector_list[0]
    pred_vector = pred_vector_list[0]
    rng = np.random.RandomState(get_hashed_seed(seed,index_process))
    sampled_indices = rng.choice(gt_vector_length, size=n_samples, replace=True, p=weights)
    several_predictions = get_if_several_predictions(pred_vector)
    score = []
    sampled_gt = gt_vector[sampled_indices]
    if several_predictions:
        for index_seed in range(len(pred_vector)):
            sampled_pred = pred_vector[index_seed][sampled_indices]
            score.append(scores_fn(sampled_gt, sampled_pred))
        score = sum(score)/len(score)
    else:
        sampled_pred = pred_vector[sampled_indices]
        score = scores_fn(sampled_gt, sampled_pred)
    return score

def get_bootstrap(scores_fn, gt_vector, pred_vector, weights = None, n_samples = None, seed = None):
    bootstrapped_scores = []
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    if n_samples is None:
        n_samples = gt_vector.shape[0]
    
    gt_vector_length = gt_vector.shape[0]

    bootstrapped_scores = Parallel(n_jobs=16, batch_size = 1, require='sharedmem')(
        delayed(bootstrap_iteration)([gt_vector], [pred_vector], scores_fn, [weights], n_samples, gt_vector_length, seed, index_iteration) 
        for index_iteration in range(n_iterations)
    )

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
    median = sorted_scores[int(0.5 * len(sorted_scores))]
    average = sorted_scores.mean()
    var = sorted_scores.var()

    return median, confidence_lower, confidence_upper, average, var, np.array(bootstrapped_scores)

def flexible_concatenate(target_array, new_array):
    # If the target array is "empty" (in this case, we define "empty" as None)
    if target_array is None:
        # Return the new array as is
        return new_array
    else:
        # Concatenate along a chosen axis if the target array is not None
        # This example uses axis 0, but this can be modified
        return np.concatenate((target_array, new_array), axis=0)

def replace_folder_with_parent(file_path, new_folder):
    parent_folder = os.path.basename(os.path.dirname(file_path))  # Get the immediate parent folder
    filename = os.path.basename(file_path)  # Get the original filename
    new_filename = f"{parent_folder}_{filename}"  # Append the parent folder to the filename
    return os.path.join(new_folder, new_filename)  # Create the new path

def hash_array(arr):
    arr = arr.astype('<U32')
    arr = np.ascontiguousarray(arr)
    arr_bytes = arr.tobytes()  # Convert to bytes
    return hashlib.sha256(arr_bytes).hexdigest()  # Get hash

def get_items_from_argparse(args, argparse_string, datasets):
    # Create an empty dictionary to store the entries
    result_dict = {}

    # Iterate through the arguments
    for arg_name, arg_value in vars(args).items():
        # Check if the argument starts with 'prediction_file_'
        if arg_name.startswith(argparse_string):
            print(arg_name, arg_value)
            # Split the argument name by '_' and extract the keys
            keys = arg_name.split('_')[2:]
            # Get the parent dictionary for the keys
            dataset = keys[0]
            if len(keys)>1:
                labeler = keys[1]
            else:
                labeler = None
            if dataset in datasets:
                parent_dict = result_dict
                for key in keys[:-1]:
                    # If the key is not in the parent dictionary, create an empty dictionary for it
                    if key not in parent_dict:
                        parent_dict[key] = {}
                    # Move to the next level in the dictionary hierarchy
                    parent_dict = parent_dict[key]
                # Assign the argument value to the dictionary with the extracted keys
                arg_value_copy = arg_value

                if arg_value is not None and arg_value!='' and (dataset=='amos' or labeler is not None):
                    from global_ import organ_denominations as organs
                    
                    converted_name = arg_value

                    if '/parsing_' in converted_name: 
                        converted_name = replace_folder_with_parent(converted_name.replace('/parsing_', '/converted_parsing_'), args.output_folder)
                        if not os.path.exists(converted_name):
                            from convert_output_format import convert_csv_format as convert_output_format
                            output_df = convert_output_format(arg_value, organs)
                            output_df.to_csv(converted_name, index=False)
                    
                    joined_organs = ["spleen",
                            "liver",
                            "kidney",
                            "stomach",
                            "pancreas",
                            "gallbladder",
                            "intestine"]
                    
                    if args.sarleset=='':
                        converted_abnormality_name = replace_folder_with_parent(converted_name.replace('.csv', '_grouped_abnormality.csv'), args.output_folder)
                        if not os.path.exists(converted_abnormality_name):
                            df = pd.read_csv(converted_name)
                            from convert_type_abnormality_to_abnormality import convert_csv_format as convert_type_abnormality_to_abnormality
                            output_df = convert_type_abnormality_to_abnormality(df, organs, dataset)
                            output_df.to_csv(converted_abnormality_name, index=False)
                        arg_value = converted_abnormality_name
                    if args.sarleset=='sarle':
                        converted_abnormality_name_sarle = replace_folder_with_parent(converted_name.replace('.csv', '_grouped_abnormality_sarle.csv'), args.output_folder)
                        if not os.path.exists(converted_abnormality_name_sarle):
                            df = pd.read_csv(converted_name)
                            from convert_type_abnormality_to_abnormality import convert_csv_format as convert_type_abnormality_to_abnormality
                            output_df = convert_type_abnormality_to_abnormality(df, joined_organs if labeler=='sarle' else organs, dataset, 'sarle')
                            output_df.to_csv(converted_abnormality_name_sarle, index=False)
                        arg_value = converted_abnormality_name_sarle
                    if args.sarleset=='other':
                        converted_abnormality_name_sarle_other = replace_folder_with_parent(converted_name.replace('.csv', '_grouped_abnormality_sarleother.csv'), args.output_folder)
                        if not os.path.exists(converted_abnormality_name_sarle_other):
                            df = pd.read_csv(converted_name)
                            from convert_type_abnormality_to_abnormality import convert_csv_format as convert_type_abnormality_to_abnormality
                            output_df = convert_type_abnormality_to_abnormality(df, joined_organs if labeler=='sarle' else organs, dataset, 'other')
                            output_df.to_csv(converted_abnormality_name_sarle_other, index=False)
                        arg_value = converted_abnormality_name_sarle_other
                    if (args.abnormality_types =='joinedorgans' or len(args.sarleset)>0) and labeler!='sarle':
                        converted_organ_name = replace_folder_with_parent(arg_value.replace('.csv', '_joinedorgans.csv'), args.output_folder)
                        
                        if not os.path.exists(converted_organ_name):
                            if args.abnormality_types =='joinedorgans':
                                from convert_join_organs import convert_csv_format as convert_join_organs
                            else:
                                from convert_join_organs_suffixes import convert_csv_format as convert_join_organs
                            convert_join_organs(arg_value, converted_organ_name, joined_organs)
                        arg_value = converted_organ_name
                    if dataset=='amos' and 'groundtruth' in argparse_string:
                        from majority_vote_amos import main as majority_vote_amos
                        directory, filename = os.path.split(arg_value)
                        new_filename = "majority_vote_" + filename

                        # Create new file path
                        majority_vote_file = replace_folder_with_parent(os.path.join(directory, new_filename), args.output_folder)
                        if not os.path.exists(majority_vote_file):
                            print(arg_value)
                            majority_vote_df = majority_vote_amos(arg_value)
                            majority_vote_df.to_csv(majority_vote_file, index=False)
                        arg_value = majority_vote_file
                parent_dict[keys[-1]] = arg_value
                arg_value = arg_value_copy
    return result_dict

def join_abnormality_name(organ, abnormality):
    return organ if abnormality=='' else organ + '_' + abnormality

def main(args):
    
    os.makedirs(args.output_folder, exist_ok=True)
    #labels
    organ_denominations =  ["spleen",
                                "liver",
                                "right kidney",
                                "left kidney",
                                "stomach",
                                "pancreas",
                                "gallbladder",
                                "small bowel", "large bowel",]
    if args.abnormality_types == 'joinedorgans' or len(args.sarleset)>0:
        abnormality_denominations =  ["spleen",
                            "liver",
                            "kidney",
                            "stomach",
                            "pancreas",
                            "gallbladder",
                            "intestine",]
    if args.abnormality_types == 'organs':
        abnormality_denominations = organ_denominations
    if args.abnormality_types == 'joinedtypes' and (args.sarleset=='sarle' or args.sarleset=='other'):
        finding_type_denominations = ['enlarged_atrophy_diffuse_focal', 'postsurgical_absent']
        abnormality_denominations = [f'{organ}_{finding_type}' for organ in organ_denominations for finding_type in finding_type_denominations]
    elif args.abnormality_types == 'joinedtypes':
        if args.type_annotation!='classifier':
            finding_type_denominations = ['enlarged_atrophy','diffuse','focal','device', 'postsurgical_absent']
        else:
            finding_type_denominations =  ["postsurgical_absent","quality","anatomy","enlarged_atrophy","device","diffuse","focal"]
        abnormality_denominations = [f'{organ}_{finding_type}' for organ in organ_denominations for finding_type in finding_type_denominations]
    abnormalities = abnormality_denominations
    def get_macro_labeler(labeler):
        macro_labeler = {'0':'human', '1':'human','2':'human','3':'human','4':'human', '5':'human'}
        if labeler in macro_labeler:
            return macro_labeler[labeler]
        return labeler
    extra_name = ''
    model_to_compare_against = ['llm']

    if (args.type_annotation=='labels'):
        
        if args.abnormality_types in ['joinedorgans','organs']:
            _labelers = ['llm','maplezqwen2']
            if len(args.sarleset)>0:
                _labelers.append('sarle')

            datasets = ['amos']

        else:
            _labelers = ['llm', 'maplezqwen2']
            if len(args.sarleset)>0:
                _labelers.append('sarle')
            datasets = ['amos']
        if args.labeler_set=='human':
            _labelers = ['llm', '0', '1','2','3','4', '5']
        output_filename = f'{args.output_folder}/label_{args.abnormality_types}_table_raw_v2.csv'
        score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score, 'specificity': specificity_score, 'mcc':matthews_corrcoef}
        word_type = 'labeling'

    if (args.type_annotation=='urgency'):
        _labelers = ['llm', '0', '1','2','3','4']
        datasets = ['amos']
        output_filename = f'{args.output_folder}/urgency_{args.abnormality_types}_table_raw_v2.csv'
        score_type_dict = {'kendalltau':kendalltau_metric}
        word_type = 'urgency'

    if (args.type_annotation=='classifier'):
        _labelers = {'llm':['test_1',
            'test_2',
            'test_3',
            'test_4',
            'test_5']}
        model_to_compare_against = ['llm']
        datasets = ['amos']
        score_type_dict = {'auc':auc_metric}
        output_filename = f'{args.output_folder}/classifier_{args.abnormality_types}_table_raw_v2.csv'
        word_type = 'classifier'
        def build_from_annotation_files(labeler, dataset):
            folder_test_files = './classifier/logs/'
            to_return = {}
            for folder_name in _labelers[labeler]:
                
                full_folder_name = find_full_folder_name(folder_test_files, folder_name)
                for index_organ, organ in enumerate(organ_denominations):
                    for index_type, type_finding in enumerate(finding_type_denominations):
                        pred_annot_filename = folder_test_files + full_folder_name + '/outputs_' + str(index_organ) + '_' + str(index_type) + '_model_outputs.csv'
                        print(pred_annot_filename)
                        loaded_predictions_annots = pd.read_csv(pred_annot_filename)
                        this_annot = (loaded_predictions_annots['annot'].values != 0)*1
                        this_pred = loaded_predictions_annots['pred'].values
                        abnormality = f'{organ}_{type_finding}'
                        ids = np.array(list(range(len(this_annot)))).astype(float)
                        if abnormality in to_return:
                            assert((this_annot==to_return[abnormality][1]).all())
                            to_return[abnormality] = (ids, this_annot, \
                                to_return[abnormality][2] + [this_pred])
                        else:
                            to_return[abnormality] = (ids, this_annot, [this_pred])
            return to_return
        
    if (args.type_annotation=='ablation_labels'):
        _labelers = ['llm', 'llmnocot', 'llmindividual', 'llmmapleztree', 'llmfast', 'llmnosplit', 'llmllama33', 'llmqwen25']
        datasets = ['amos']
        output_filename = f'{args.output_folder}/ablation_labels_{args.abnormality_types}_table_raw_v2.csv'
        score_type_dict = {'f1':f1_score, 'precision': precision_score, 'recall': recall_score, 'specificity': specificity_score, 'mcc':matthews_corrcoef}
        word_type = 'ablation_labels'
    if (args.type_annotation=='ablation_urgency'):
        _labelers = ['llm', 'llmnocot', 'llmindividual', 'llmmapleztree', 'llmfast', 'llmnosplit', 'llmllama33', 'llmqwen25']
        datasets = ['amos']
        output_filename = f'{args.output_folder}/ablation_urgency_{args.abnormality_types}_table_raw_v2.csv'
        score_type_dict = {'kendalltau':kendalltau_metric}
        word_type = 'ablation_urgency'

    prediction_files_dict = get_items_from_argparse(args, 'prediction_file_', datasets)
    groundtruth_csv_dict = get_items_from_argparse(args, 'groundtruth_csv_', datasets)
    bylabeler_csv_dict = get_items_from_argparse(args, 'bylabeler_csv_', datasets)
    aggregation_datasets = copy.copy(datasets)
    if args.do_macro_scores:
        aggregation_datasets += [f'{dataset_name}_macro' for dataset_name in aggregation_datasets]

    def labelers(dataset):
        if args.labeler_set=='llm':
            return ['llm']
        to_return = []
        for labeler in _labelers:
            to_return.append(labeler)
        return to_return

    results = {}
    bootstrapped_scores = {}
    for dataset in datasets:
        results[dataset] = {}

        groundtruth = groundtruth_csv_dict[dataset]
        bylabeler = bylabeler_csv_dict[dataset]
        dataset_arg = dataset
        list_of_labelers = labelers(dataset)
        llm_prediction = {}
        for labeler in list_of_labelers:
            if not args.type_annotation=='classifier':
                results[dataset][labeler] = {}
                done_before = False
                for abnormality in abnormalities:
                    file_path = Path(f'{args.output_folder}/{abnormality[:100]}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv')
                    done_before = done_before or file_path.exists()
                if not done_before:
                    args2 = SimpleNamespace(type_annotation = 'urgency' if ('urgency' in args.type_annotation) else 'labels', \
                        groundtruth = groundtruth, labeler = labeler, dataset = dataset_arg,
                        single_file = prediction_files_dict[dataset_arg][get_macro_labeler(labeler)],
                        abnormalities = abnormalities
                        )
                    if args.labeler_set=='human' and not (labeler in model_to_compare_against):
                        args2.exclude_labeler = int(labeler)
                        if bylabeler is not None:
                            args2.groundtruth = bylabeler
                    results[dataset][labeler] = get_labels(args2)
                    print('oi11',args2, results[dataset][labeler])
                    for abnormality in abnormalities:
                        if (abnormality+'_annotpred') in results[dataset][labeler]:
                            ids, annots, preds = results[dataset][labeler][abnormality+'_annotpred']
                            if (len(ids)>0):
                                df = pd.DataFrame({
                                    'id': ids,
                                    'annot': annots,
                                    'pred': preds
                                })
                                csv_file_path = f'{args.output_folder}/{abnormality[:100]}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv' 
                                df.to_csv(csv_file_path, index=False)  
                else:
                    for abnormality in abnormalities:
                        file_path = Path(f'{args.output_folder}/{abnormality[:100]}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv')
                        if file_path.exists():
                            csv_file_path = f'{args.output_folder}/{abnormality[:100]}_{labeler}_{dataset}_{word_type}_outputs{extra_name}.csv' 
                            df = pd.read_csv(csv_file_path)
                            # Convert DataFrame columns to numpy arrays
                            id_array = df['id'].to_numpy().astype(str)
                            annot_array = df['annot'].to_numpy()
                            pred_array = df['pred'].to_numpy()
                            if 'urgency' in args.type_annotation:
                                if labeler=='llm':
                                    llm_prediction[abnormality] = id_array, pred_array
                                    pred_array_llm = pred_array
                                    id_array_llm = id_array
                                else:
                                    id_array_llm, pred_array_llm = llm_prediction[abnormality] 
                                
                                idx = np.searchsorted(id_array, id_array_llm)
                                mask2 = np.zeros_like(id_array_llm, dtype=bool)  # Initialize mask
                                valid_idx = idx < len(id_array)  # Ensure indices are in bounds
                                mask2[valid_idx] = id_array[idx[valid_idx]] == id_array_llm[valid_idx]

                                id_array_llm = id_array_llm[mask2]
                                pred_array_llm = pred_array_llm[mask2]
                                print(csv_file_path, len(id_array), len(id_array_llm))
                                assert(len(id_array) == len(id_array_llm))
                                pred_array_mask = (pred_array>=0)
                                annot_array_mask = (annot_array>=0)
                                pred_array_llm_mask = (pred_array_llm>=0)
                                both_masks = annot_array_mask & pred_array_mask & pred_array_llm_mask
                                id_array = id_array[both_masks]
                                annot_array = annot_array[both_masks]
                                pred_array = pred_array[both_masks]

                            results[dataset][labeler][abnormality] = id_array, annot_array, pred_array
            else:
                results[dataset][labeler] = build_from_annotation_files(labeler, dataset)
    file_path_table = Path(output_filename)
    # if file_path_table.exists():
    #     df_table = pd.read_csv(output_filename)
    # else:
    df_table = pd.DataFrame()
    for abnormality in abnormalities:
        print(abnormality)
        for dataset in datasets:
            print(dataset)
            rng_seed = random.randint(0, 2**32 - 1)
            for labeler in labelers(dataset):
                row = {'abnormality': abnormality, 'dataset':dataset, 'labeler':labeler, 'n':None, 'n_pos':None}
                # values_to_check = {'abnormality': [abnormality], 'dataset':[dataset],'labeler':[labeler]}
                # if 'abnormality' in df_table.columns and 'dataset' in df_table.columns and 'labeler' in df_table.columns and df_table.loc[:, list(values_to_check.keys())].isin(values_to_check).all(axis=1).any():
                #     continue
                print(labeler)
                if abnormality in results[dataset][labeler]:
                    id_vector, gt_vector,pred_vector = results[dataset][labeler][abnormality] 
                    hashed_ids = hash_array(id_vector)
                    if 'urgency' in args.type_annotation:
                        most_common_gt = mode(gt_vector).mode
                        n_pos =((~np.isclose(gt_vector,most_common_gt))).sum()
                    else:
                        n_pos = (gt_vector>0).sum()
                    
                    if n_pos<=10:
                        continue
                    row['n_pos'] = n_pos
                    row['n'] = len(gt_vector)
                    for score in score_type_dict:
                        row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average'], _, sampled_scores  = get_bootstrap(score_type_dict[score], gt_vector, pred_vector, seed = rng_seed)
                        if labeler not in bootstrapped_scores:
                            bootstrapped_scores[labeler] = {}
                        if score not in bootstrapped_scores[labeler]:
                            bootstrapped_scores[labeler][score] = {}
                        bootstrapped_scores[labeler][score][abnormality + '_' + dataset + '_' + hashed_ids] = sampled_scores, row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average']
                    
                    for labeler2 in model_to_compare_against:
                        if labeler == labeler2:
                            continue
                        if labeler2 in results[dataset]:

                            id_vector2, gt_vector2,pred_vector2 = results[dataset][labeler2][abnormality] 
                            print(len(id_vector), len(id_vector2))
                            #getting only the ids from id_vector2 that are in id_vector
                            idx = np.searchsorted(id_vector, id_vector2)
                            mask2 = np.zeros_like(id_vector2, dtype=bool)  # Initialize mask
                            valid_idx = idx < len(id_vector)  # Ensure indices are in bounds
                            mask2[valid_idx] = id_vector[idx[valid_idx]] == id_vector2[valid_idx]

                            id_vector2 = id_vector2[mask2]
                            gt_vector2 = gt_vector2[mask2]
                            pred_vector2 = pred_vector2[mask2]
                            print(len(id_vector), len(id_vector2))
                            assert(len(id_vector)==len(id_vector2))

                            for score in score_type_dict:
                                if not ((abnormality + '_' + dataset + '_' + hashed_ids) in bootstrapped_scores[labeler2][score]):
                                    row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'], _, sampled_scores2  = get_bootstrap(score_type_dict[score], gt_vector2, pred_vector2, seed = rng_seed)

                                    bootstrapped_scores[labeler2][score][abnormality + '_' + dataset + '_' + hashed_ids] = sampled_scores2,  row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average']
                                else:
                                    sampled_scores2,  row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'] =  bootstrapped_scores[labeler2][score][abnormality + '_' + dataset + '_' + hashed_ids]

                                row[f'{get_macro_labeler(labeler2)}_{score}_p'] = get_hypothesis_test_equality(sampled_scores, sampled_scores2, row[f'{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_median'])

                if row['n_pos'] is not None:
                    df_table = pd.concat([df_table, pd.DataFrame(row, index=[0])], ignore_index=True)
                    df_table.to_csv(output_filename, index=False)
    combinations = []
    if args.type_annotation!='classifier':
        combinations = []
        for aggregate_dataset in range(2):  # 0 or 1
            for aggregate_organ in range(2):  # 0 or 1
                for aggregate_abnormality in range(2):  # 0 or 1
                    for aggregate_labeler in range(2):  # 0 or 1
                    
                        # Add your conditions here
                        # if not aggregate_dataset and not aggregate_abnormality and not aggregate_labeler:
                        #     continue
                        if args.abnormality_types in ['joinedorgans', 'organs'] and aggregate_abnormality:
                            continue
                        if args.labeler_set!='human' and aggregate_labeler:
                            continue
                        if len(datasets)==1 and aggregate_dataset:
                            continue
                        combinations.append({'dataset':aggregate_dataset, 'organ': aggregate_organ, 'abnormality': aggregate_abnormality, 'labeler': aggregate_labeler})
    else:
        combinations = [{'dataset':0, 'organ': 0, 'abnormality': 0, 'labeler': 0}]

    #TEMP
    # combinations = [{'dataset':1, 'organ': 1, 'abnormality': 1, 'labeler': 0}]

    print('oi4', combinations)

    #grouped by label or dataset
    # print(df_table)
    df_table['aggregation'] = False
    vector_sum = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    count_n = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    print(df_table['abnormality'])
    # df_table[['organ', 'abnormality']] = df_table['abnormality'].str.split('_', n=1, expand=True)
    split_cols = df_table['abnormality'].str.split('_', n=1, expand=True)
    df_table['organ'] = split_cols[0]
    df_table['abnormality'] = split_cols[1].fillna('') if 1 in split_cols.columns else ''


    cols = ['organ', 'abnormality'] + [c for c in df_table.columns if c not in ['organ', 'abnormality']]

    # Reorder and save to CSV
    df_table = df_table[cols]
    organs = pd.unique(df_table['organ'])
    abnormalities_after_split = pd.unique(df_table['abnormality'])

    for combination in combinations:
        for dataset in datasets:
            for organ in organs:
                for abnormality in abnormalities_after_split:
                    rng_seed = random.randint(0, 2**32 - 1)
                    for labeler in labelers(dataset):
            
                        if not combination['abnormality']:
                            this_df_table = df_table[df_table['abnormality']==abnormality]
                        else:
                            abnormality = 'micro_ag'
                            this_df_table = df_table
                        if not combination['organ']:
                            this_df_table = this_df_table[this_df_table['organ']==organ]
                        else:
                            organ = 'micro_ag'
                        if not combination['labeler']:
                            this_df_table = this_df_table[this_df_table['labeler']==labeler]
                        else:
                            labeler = 'micro_ag'
                            this_df_table = this_df_table[~this_df_table['labeler'].isin(model_to_compare_against)]
                            this_df_table = this_df_table[this_df_table['labeler']!='llm']
                        if not combination['dataset']:
                            this_df_table = this_df_table[this_df_table['dataset']==dataset]
                        else:
                            dataset = 'micro_ag'
                        this_df_table = this_df_table[this_df_table['aggregation']!=True]
                        this_df_table = this_df_table[this_df_table['n_pos']>10]
                        row = {'dataset': dataset, 'abnormality': abnormality, 'organ': organ, 'labeler': labeler, 'aggregation': True}

                        # values_to_check = {'abnormality': [row['abnormality']], 'organ': [row['organ']], 'dataset':[row['dataset']], 'labeler':[row['labeler']], 'aggregation':[True]}
                        # if df_table.loc[:, list(values_to_check.keys())].isin(values_to_check).all(axis=1).any():
                        #     continue
                        if len(this_df_table)>=1:
                            if not args.type_annotation in ['ablation_urgency','urgency','classifier']:
                                row['n'] = this_df_table['n'].sum()
                                row['n_pos'] = this_df_table['n_pos'].sum()
                                for score in score_type_dict:
                                    id_vector = np.array([])
                                    gt_vector = np.array([])
                                    dataset_sizes = np.array([])
                                    pred_vector = None
                                    for _, old_row in this_df_table.iterrows():
                                        this_id_vector, this_gt_vector,this_pred_vector = results[old_row['dataset']][old_row['labeler']][join_abnormality_name(old_row['organ'], old_row['abnormality'])] 

                                        # hashed_ids = hash_array(this_id_vector)

                                        dataset_size = len(this_gt_vector)
                                        dataset_sizes = np.hstack((dataset_sizes, [dataset_size]))
                                        gt_vector = np.hstack((gt_vector,this_gt_vector))
                                        id_vector = np.hstack((id_vector,this_id_vector))
                                        if isinstance(this_pred_vector,list):
                                            for index_preds in range(len(this_pred_vector)):
                                                if pred_vector is None:
                                                    pred_vector = []
                                                if pred_vector is not None and len(pred_vector)<index_preds+1:
                                                    pred_vector.append(None)
                                                pred_vector[index_preds] = flexible_concatenate(pred_vector[index_preds],this_pred_vector[index_preds])
                                        else:
                                            pred_vector = flexible_concatenate(pred_vector,this_pred_vector)
                                    hashed_ids = hash_array(id_vector)
                                    if len(this_df_table)==1:
                                        row2 = this_df_table.iloc[0].to_dict()
                                        row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average'] = row2[f'{score}_median'], row2[f'{score}_low'], row2[f'{score}_high'], row2[f'{score}_average']
                                        sampled_scores = bootstrapped_scores[row2['labeler']][score][join_abnormality_name(row2['organ'], row2['abnormality']) + '_' + row2['dataset'] + '_' + hashed_ids][0]
                                    else:
                                        row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average'], _, sampled_scores = get_bootstrap(score_type_dict[score], gt_vector, pred_vector, None, seed = rng_seed)
                                    if labeler not in bootstrapped_scores:
                                        bootstrapped_scores[labeler] = {}
                                    if score not in bootstrapped_scores[labeler]:
                                        bootstrapped_scores[labeler][score] = {}
                                    bootstrapped_scores[labeler][score][join_abnormality_name(organ, abnormality) + '_' + dataset + '_' + hashed_ids] = sampled_scores, row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average']
                                    if args.do_macro_scores and (labeler not in model_to_compare_against or args.labeler_set=='llm'):
                                        vector_sum[frozenset(combination.items())][score]['main'] += sampled_scores
                                        count_n[frozenset(combination.items())][score]['main'] += 1
                                    for labeler2 in model_to_compare_against:
                                        if labeler == labeler2:
                                            continue
                                        id_vector2 = np.array([])
                                        gt_vector2 = np.array([])
                                        pred_vector2 = None
                                        start_indx_dataset_size = 0
                                        count_index_table_row = 0
                                        for _, old_row in this_df_table.iterrows():
                                            this_id_vector, this_gt_vector,this_pred_vector = results[old_row['dataset']][labeler2][join_abnormality_name(old_row['organ'] , old_row['abnormality'])] 

                                            filtered_id_vector = id_vector[start_indx_dataset_size: start_indx_dataset_size+int(dataset_sizes[count_index_table_row])]
                                            start_indx_dataset_size += int(dataset_sizes[count_index_table_row])
                                            count_index_table_row += 1
                                            #getting only the ids from id_vector2 that are in id_vector
                                            idx = np.searchsorted(filtered_id_vector, this_id_vector)
                                            mask2 = np.zeros_like(this_id_vector, dtype=bool)  # Initialize mask
                                            valid_idx = idx < len(filtered_id_vector)  # Ensure indices are in bounds
                                            mask2[valid_idx] = filtered_id_vector[idx[valid_idx]] == this_id_vector[valid_idx]

                                            this_id_vector = this_id_vector[mask2]
                                            this_gt_vector = this_gt_vector[mask2]
                                            this_pred_vector = this_pred_vector[mask2]
                                            assert(len(filtered_id_vector)==len(this_id_vector))

                                            id_vector2 = np.hstack((id_vector2,this_id_vector))
                                            gt_vector2 = np.hstack((gt_vector2,this_gt_vector))
                                            if isinstance(this_pred_vector,list):
                                                for index_preds in range(len(this_pred_vector)):
                                                    if pred_vector2 is None:
                                                        pred_vector2 = []
                                                    if pred_vector2 is not None and len(pred_vector2)<index_preds+1:
                                                        pred_vector2.append(None)
                                                    pred_vector2[index_preds] = flexible_concatenate(pred_vector2[index_preds],this_pred_vector[index_preds])
                                            else:
                                                pred_vector2 = flexible_concatenate(pred_vector2,this_pred_vector)
                                        
                                        assert(len(id_vector)==len(id_vector2))

                                        if not ((join_abnormality_name(organ, abnormality) + '_' + dataset + '_' + hashed_ids) in bootstrapped_scores[labeler2][score]):
                                            if len(this_df_table)==1:
                                                sampled_scores2, row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'] = bootstrapped_scores[labeler2][score][join_abnormality_name(row2['organ'], row2['abnormality']) + '_' + row2['dataset'] + '_' + hashed_ids]

                                                bootstrapped_scores[labeler2][score][join_abnormality_name(organ, abnormality) + '_' + dataset + '_' + hashed_ids] = sampled_scores2,  row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average']
                                            else:
                                                row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'], _, sampled_scores2  = get_bootstrap(score_type_dict[score], gt_vector2, pred_vector2, seed = rng_seed)

                                                bootstrapped_scores[labeler2][score][join_abnormality_name(organ, abnormality) + '_' + dataset + '_' + hashed_ids] = sampled_scores2,  row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average']
                                        else:
                                            sampled_scores2,  row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'] =  bootstrapped_scores[labeler2][score][join_abnormality_name(organ, abnormality) + '_' + dataset + '_' + hashed_ids]
                                        
                                        if len(this_df_table)==1:
                                            row[f'{get_macro_labeler(labeler2)}_{score}_p'] = row2[f'{get_macro_labeler(labeler2)}_{score}_p'] 
                                        else:
                                            row[f'{get_macro_labeler(labeler2)}_{score}_p'] = get_hypothesis_test_equality(sampled_scores, sampled_scores2, row[f'{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_median'])
                                        print(labeler, score, row[f'{get_macro_labeler(labeler2)}_{score}_p'])

                                        if args.do_macro_scores:
                                            vector_sum[frozenset(combination.items())][score][labeler2] += sampled_scores2
                                            count_n[frozenset(combination.items())][score][labeler2] += 1
                            else:
                                this_vector_sum = 0.
                                this_count_n = 0
                                id_vectors = []
                                for _, old_row in this_df_table.iterrows():
                                    this_id_vector, this_gt_vector,this_pred_vector = results[old_row['dataset']][old_row['labeler']][join_abnormality_name(old_row['organ'], old_row['abnormality'])] 
                                    id_vectors.append(this_id_vector)
                                    hashed_ids = hash_array(this_id_vector)
                                    sampled_scores = bootstrapped_scores[old_row['labeler']][score][join_abnormality_name(old_row['organ'], old_row['abnormality']) + '_' + old_row['dataset'] + '_' + hashed_ids][0]

                                    this_vector_sum += sampled_scores
                                    this_count_n += 1
                                if len(this_df_table)==1:
                                    row2 = this_df_table.iloc[0].to_dict()
                                    row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average'] = row2[f'{score}_median'], row2[f'{score}_low'], row2[f'{score}_high'], row2[f'{score}_average']
                                else:
                                    row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average'], _, sampled_scores = get_macro_bootstrap(this_vector_sum, this_count_n)
                                if args.do_macro_scores and (labeler not in model_to_compare_against or args.labeler_set=='llm'):
                                    vector_sum[frozenset(combination.items())][score]['main'] += sampled_scores
                                    count_n[frozenset(combination.items())][score]['main'] += 1
                                for labeler2 in model_to_compare_against:
                                    if labeler==labeler2:
                                        continue
                                    
                                    count_index_table_row = 0
                                    this_vector_sum2 = 0.
                                    this_count_n2 = 0
                                    for _, old_row in this_df_table.iterrows():
                                        this_id_vector, this_gt_vector,this_pred_vector = results[old_row['dataset']][labeler2][join_abnormality_name(old_row['organ'], old_row['abnormality'])] 

                                        filtered_id_vector = id_vectors[count_index_table_row]
                                        count_index_table_row += 1
                                        #getting only the ids from id_vector2 that are in id_vector
                                        idx = np.searchsorted(filtered_id_vector, this_id_vector)
                                        mask2 = np.zeros_like(this_id_vector, dtype=bool)  # Initialize mask
                                        valid_idx = idx < len(filtered_id_vector)  # Ensure indices are in bounds
                                        mask2[valid_idx] = filtered_id_vector[idx[valid_idx]] == this_id_vector[valid_idx]

                                        this_id_vector = this_id_vector[mask2]
                                        this_gt_vector = this_gt_vector[mask2]
                                        this_pred_vector = this_pred_vector[mask2]
                                        
                                        hashed_ids = hash_array(this_id_vector)
                                        if join_abnormality_name(old_row['organ'], old_row['abnormality']) + '_' + old_row['dataset'] + '_' + hashed_ids in bootstrapped_scores[labeler2][score]:
                                            sampled_scores = bootstrapped_scores[labeler2][score][join_abnormality_name(old_row['organ'], old_row['abnormality']) + '_' + old_row['dataset'] + '_' + hashed_ids][0]
                                        else:
                                            _, _, _, _, _, sampled_scores = get_bootstrap(score_type_dict[score], this_gt_vector, this_pred_vector, seed = rng_seed)
                                        this_vector_sum2 += sampled_scores
                                        this_count_n2 += 1
                                    row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'], _, sampled_scores2 = get_macro_bootstrap(this_vector_sum2, this_count_n2)
                                    if len(this_df_table)==1:
                                        row[f'{get_macro_labeler(labeler2)}_{score}_p'] = row2[f'{get_macro_labeler(labeler2)}_{score}_p'] 
                                    else:
                                        row[f'{get_macro_labeler(labeler2)}_{score}_p'] = get_hypothesis_test_equality_macro(this_vector_sum, this_vector_sum2, this_count_n, row[f'{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_median'])
                                    if args.do_macro_scores:
                                        vector_sum[frozenset(combination.items())][score][labeler2] += sampled_scores2
                                        count_n[frozenset(combination.items())][score][labeler2] += 1
                                                


                            if combination['organ'] or combination['abnormality'] or combination['labeler'] or combination['dataset']:
                                df_table = pd.concat([df_table, pd.DataFrame(row, index=[0])], ignore_index=True)
                                df_table.to_csv(output_filename, index=False)
                            print('oi5')
                        if combination['labeler']:
                            break
                    if combination['abnormality']:
                        break
                if combination['organ']:
                    break
            if combination['dataset']:
                break
    print(vector_sum.keys())
    if args.do_macro_scores:
        for combination in vector_sum:
            dict_combination = dict(combination)
            row = {'dataset': 'micro_ag' if dict_combination['dataset'] else 'macro_ag', 'organ': 'micro_ag' if dict_combination['organ'] else 'macro_ag', 'abnormality': 'micro_ag' if dict_combination['abnormality'] else 'macro_ag', 'labeler': 'micro_ag' if dict_combination['labeler'] else 'macro_ag', 'aggregation': True}
            only_one_row = False
            for score in vector_sum[combination]:
                if count_n[combination][score]['main']==1:
                    only_one_row = True
                    break
                row[f'{score}_median'], row[f'{score}_low'], row[f'{score}_high'], row[f'{score}_average'], _, sampled_scores = get_macro_bootstrap(vector_sum[combination][score]['main'], count_n[combination][score]['main'])

                for labeler2 in model_to_compare_against:
                    if labeler==labeler2:
                        continue
                    row[f'{get_macro_labeler(labeler2)}_{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_low'], row[f'{get_macro_labeler(labeler2)}_{score}_high'], row[f'{get_macro_labeler(labeler2)}_{score}_average'], _, sampled_scores2 = get_macro_bootstrap(vector_sum[combination][score][labeler2], count_n[combination][score][labeler2])
                    print(count_n[combination][score][labeler2], count_n[combination][score]['main'], combination)
                    assert(count_n[combination][score][labeler2]==count_n[combination][score]['main'])

                    row[f'{get_macro_labeler(labeler2)}_{score}_p'] = get_hypothesis_test_equality_macro(vector_sum[combination][score]['main'], vector_sum[combination][score][labeler2], count_n[combination][score]['main'], row[f'{score}_median'], row[f'{get_macro_labeler(labeler2)}_{score}_median'])
            if not only_one_row:
                df_table = pd.concat([df_table, pd.DataFrame(row, index=[0])], ignore_index=True)
                df_table.to_csv(output_filename, index=False)

import argparse
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
    parser.add_argument("--type_annotation", type=str, choices = ['labels', 'urgency', 'classifier', 'ablation_labels', 'ablation_urgency'], default="labels",\
        help='''The type of annotation/prediction to create a table for.''')
    parser.add_argument("--output_folder", type=str, default="./", help='''The folder where the results are going to be written to.''')

    parser.add_argument("--groundtruth_csv_amos", type=str, default='amos_test_annotations.csv', help='''''')

    parser.add_argument("--prediction_file_amos_llm", type=str, default='./results/qwen2_leavs/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_human", type=str, default='amos_test_annotations.csv', help='''''')
    parser.add_argument("--prediction_file_amos_sarle", type=str, default='converted_sarle_labels_testamos.csv', help='''''')
    parser.add_argument("--prediction_file_amos_maplezqwen2", type=str, default='./results/maplez_qwen2_amostest/parsing_results_llm.csv', help='''''')

    parser.add_argument("--prediction_file_amos_llmnocot", type=str, default='./results/ablation_nocot_amostest/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_llmindividual", type=str, default='./results/ablation_individual_amostest/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_llmmapleztree", type=str, default='./results/ablation_mapleztree_amostest/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_llmfast", type=str, default='./results/ablation_fast_amostest/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_llmnosplit", type=str, default='./results/ablation_no_amostest/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_llmllama33", type=str, default='./results/ablation_llama33_amostest/parsing_results_llm.csv', help='''''')
    parser.add_argument("--prediction_file_amos_llmqwen25", type=str, default='./results/ablation_qwen25_amostest/parsing_results_llm.csv', help='''''')

    parser.add_argument("--labeler_set", type=str, default='benchmark', choices = ['llm','human','benchmark'], help='''llm: get results only for the leavs method. human: get comparison against humans. benchmark: get comparison against maplez and sarle (if sarleset is set)''')
    parser.add_argument("--abnormality_types", type=str, choices = ['joinedtypes', 'joinedorgans', 'organs'], default="joinedorgans",\
        help='''joinedtypes: calculate results for finding types and organs. organs: calculate results for any abnromality the organs after joining predictions and ground truths for finding types. joinedorgans: join kidneys and intestine into one organ and calculate results for any abnormality type in the organ. ''')
    parser.add_argument("--sarleset", type=str, choices = ['', 'sarle', 'other'], default="",\
        help='''If you would like to include sarle in the benchmark comparison, set it to either sarle or other. With other, the Other label from sarle outputs will be considered an abnorality. Otherwise it won't.''')
    
    args = parser.parse_args()
    args.do_macro_scores = True
    if args.labeler_set=='human':
        args.bylabeler_csv_amos = args.groundtruth_csv_amos
    else:
        args.bylabeler_csv_amos = ''
    
    main(args)
