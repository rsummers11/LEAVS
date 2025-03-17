# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

#class used to manage the accumulation and calculation of metrics
import collections
from sklearn.metrics import roc_auc_score
import numpy as np
import time
import torch
from skimage import measure
import nibabel as nib
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import math

def optimal_f1_threshold(y_true, y_scores):

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Compute F1 score for each threshold
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-10)  # Avoid division by zero
    optimal_idx = np.argmax(f1_scores)
    
    return thresholds[optimal_idx]

def youden_index(y_true, y_scores):
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Compute Youden's J statistic
    youden_j = tpr - fpr
    optimal_idx = np.argmax(youden_j)  # Index of max J
    
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def f1_score(ground_truth, predictions):
    true_positive = (predictions * ground_truth).sum().item()
    false_positive = ((1 - ground_truth) * predictions).sum().item()
    false_negative = (ground_truth * (1 - predictions)).sum().item()
    true_negative = ((1-predictions) * (1-ground_truth)).sum().item()

    if true_positive == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    precision = true_positive / (true_positive + false_positive + 1e-20)
    recall = true_positive / (true_positive + false_negative + 1e-20)
    mcc = ((true_positive*true_negative)-(false_negative*false_positive))/(math.sqrt((true_positive+false_positive)*(true_positive+false_negative)*(true_negative+false_positive)*(true_negative+false_negative))+ 1e-20)
    specificity = true_negative/(true_negative+false_positive+ 1e-20)
    if precision + recall == 0:
        return 0.0, 0.0, 0.0

    return 2 * (precision * recall) / (precision + recall + 1e-20), precision, recall, specificity, mcc

from sklearn.metrics import auc
def calculate_auc_curve(thresholds, function):
    tpr = []
    fpr = []


    for threshold in thresholds:
        false_positives, true_positives, false_negatives, true_negatives = function(threshold)
        
        true_positive_rate = true_positives / (true_positives + false_negatives)
        false_positive_rate = false_positives / (false_positives + true_negatives)
        
        tpr.append(true_positive_rate)
        fpr.append(false_positive_rate)
        
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr

class Metrics():
    def __init__(self, args):
        self.values = collections.defaultdict(list)
        self.start_times = {}
        self.registered_predictions = {}
        self.normal_abnormal = args.normal_abnormal
    
    #methods used to measure how much time differentaprts of the code take
    def start_time(self, key):
        self.start_times[key] = time.time()
    
    def end_time(self, key):
        self.values['time_' + key] += [time.time()- self.start_times[key]]
    
    def add_list(self, key, value):
        value = value.detach().cpu()
        self.values[key].append(value)
    
    # add metrics that have its average calculate by simply averaging all the values given during one epoch
    # loss is an example of a value that can use this function
    # DO NOT use this function for models predictions (used for calculating AUC) and for iou values
    def add_value(self, key, value):
        value = value.detach().cpu().item()
        self.values[key].append( value)
    
    def calculate_average(self):
        self.average = {}       
        
        for key, element in self.values.items():         
            #calculate the average for all other keys
            n_values = len(element)
            if n_values == 0:
                self.average[key] = 0
                continue
            sum_values = sum(element)
            self.average[key] = sum_values/float(n_values)
            if 'y_predicted_' in key:
                from global_ import organ_denominations
                from global_ import finding_types
                if self.normal_abnormal is not None:
                    finding_types = ['abnormal']
                else:
                    from global_ import finding_types
                groundtruths_thresholded_all_organs = None
                predictions_thresholded_all_organs = None
                predictions_notthresholded_all_organs = None
                auc_scores = []
                for index_organ, organ in enumerate(organ_denominations):
                    predictions_thresholded = None
                    predictions_notthresholded = None
                    groundtruths_thresholded = None
                    for index_finding_type, finding_type in enumerate(finding_types):
                        ground_truth = torch.cat(self.values[key.replace('y_predicted_','y_true_')],0)[:,index_organ,index_finding_type].reshape(-1)
                        prediction = torch.cat(self.values[key],0)[:,index_organ,index_finding_type]
                        filter = torch.cat(self.values[key.replace('y_predicted_','y_filter_')],0)[:,index_organ,index_finding_type].reshape(-1).bool()
                        ground_truth = ground_truth[filter]
                        prediction = prediction[filter]
                        if organ=='liver':
                            print('oi12', ground_truth.sum(), len(ground_truth))
                        if ground_truth.sum()>10:
                            best_threshold = optimal_f1_threshold(ground_truth, prediction)
                            prediction_thresholded = ((prediction>best_threshold)*1)
                            self.average[f'best_threshold_{organ}_{finding_type}'] = best_threshold
                            if predictions_thresholded is None:
                                predictions_thresholded = prediction_thresholded
                                groundtruths_thresholded = ground_truth
                                predictions_notthresholded = prediction
                            else:
                                predictions_thresholded = torch.cat((predictions_thresholded, prediction_thresholded), axis=0)
                                predictions_notthresholded = torch.cat((predictions_notthresholded, prediction), axis=0)
                                groundtruths_thresholded = torch.cat((groundtruths_thresholded, ground_truth), axis=0)
                            self.average[key.replace('y_predicted_', f'f1_score_{organ}_{finding_type}_')], \
                                self.average[key.replace('y_predicted_', f'precision_{organ}_{finding_type}_')], \
                                self.average[key.replace('y_predicted_', f'recall_{organ}_{finding_type}_')],\
                                self.average[key.replace('y_predicted_', f'specificity_{organ}_{finding_type}_')],\
                                self.average[key.replace('y_predicted_', f'mcc_{organ}_{finding_type}_')] = f1_score(prediction_thresholded.reshape(-1), ground_truth)
                            if ground_truth.sum()!=len(ground_truth):
                                auc_score = roc_auc_score(ground_truth, prediction.reshape(-1))
                                self.average[key.replace('y_predicted_', f'auc_score_{organ}_{finding_type}_')] = auc_score
                                auc_scores.append(auc_score)
                            for threshold in [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5]:
                                self.average[key.replace('y_predicted_', f'f1_score_{organ}_{finding_type}_{threshold}')], \
                                    self.average[key.replace('y_predicted_', f'precision_{organ}_{finding_type}_{threshold}')], \
                                    self.average[key.replace('y_predicted_', f'recall_{organ}_{finding_type}_{threshold}')],\
                                    self.average[key.replace('y_predicted_', f'specificity_{organ}_{finding_type}_{threshold}')], \
                                    self.average[key.replace('y_predicted_', f'mcc_{organ}_{finding_type}_{threshold}')] = f1_score(((prediction>threshold)*1).reshape(-1), ground_truth)

                    if predictions_thresholded is not None:
                        if predictions_thresholded_all_organs is None:
                            predictions_thresholded_all_organs = predictions_thresholded.reshape(-1)
                            predictions_notthresholded_all_organs = predictions_notthresholded.reshape(-1)
                            groundtruths_thresholded_all_organs = groundtruths_thresholded.reshape(-1)
                        else:
                            predictions_thresholded_all_organs = torch.cat((predictions_thresholded_all_organs, predictions_thresholded.reshape(-1)), axis=0)
                            groundtruths_thresholded_all_organs = torch.cat((groundtruths_thresholded_all_organs, groundtruths_thresholded.reshape(-1)), axis=0)
                            predictions_notthresholded_all_organs = torch.cat((predictions_notthresholded_all_organs, predictions_notthresholded.reshape(-1)), axis=0)


                        self.average[key.replace('y_predicted_', f'f1_score_{organ}_')], \
                            self.average[key.replace('y_predicted_', f'precision_{organ}_')], \
                            self.average[key.replace('y_predicted_', f'recall_{organ}_')], \
                            self.average[key.replace('y_predicted_', f'specificity_{organ}_')], \
                            self.average[key.replace('y_predicted_', f'mcc_{organ}_')] = f1_score(predictions_thresholded.reshape(-1), groundtruths_thresholded.reshape(-1))
                        for threshold in [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5]:
                            self.average[key.replace('y_predicted_', f'organ_f1_score_{organ}_{threshold}')], \
                                self.average[key.replace('y_predicted_', f'organ_precision_{organ}_{threshold}')], \
                                self.average[key.replace('y_predicted_', f'organ_recall_{organ}_{threshold}')], \
                                self.average[key.replace('y_predicted_', f'organ_specificity_{organ}_{threshold}')], \
                                self.average[key.replace('y_predicted_', f'organ_mcc_{organ}_{threshold}')] = f1_score(((predictions_notthresholded>threshold)*1).reshape(-1), groundtruths_thresholded.reshape(-1))

                ground_truth = torch.cat(self.values[key.replace('y_predicted_','y_true_')],0).view(-1)
                prediction = torch.cat(self.values[key],0).view(-1)
                filter = torch.cat(self.values[key.replace('y_predicted_','y_filter_')],0).view(-1).bool()
                ground_truth = ground_truth[filter]
                prediction = prediction[filter]
                self.average[key.replace('y_predicted_', f'all_outputs_f1_score_')], \
                    self.average[key.replace('y_predicted_', f'all_outputs_precision_')], \
                    self.average[key.replace('y_predicted_', f'all_outputs_recall_')], \
                    self.average[key.replace('y_predicted_', f'all_outputs_specificity_')], \
                    self.average[key.replace('y_predicted_', f'all_outputs_mcc_')] = f1_score(predictions_thresholded_all_organs.view(-1), groundtruths_thresholded_all_organs.reshape(-1))                # self.average[key.replace('y_predicted_', 'all_outputs_auc_score_')] = roc_auc_score(ground_truth, torch.cat(self.values[key],0).view(-1))
                for threshold in [-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5]:
                    self.average[key.replace('y_predicted_', f'all_outputs_f1_score_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'all_outputs_precision_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'all_outputs_recall_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'all_outputs_specificity_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'all_outputs_mcc_{threshold}')], = f1_score(((prediction>threshold)*1) , ground_truth)                # self.average[key.replace('y_predicted_', 'all_outputs_auc_score_')] = roc_auc_score(ground_truth, torch.cat(self.values[key],0).view(-1))
                    self.average[key.replace('y_predicted_', f'filtered_outputs_f1_score_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'filtered_outputs_precision_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'filtered_outputs_recall_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'filtered_outputs_specificity_{threshold}')], \
                        self.average[key.replace('y_predicted_', f'filtered_outputs_mcc_{threshold}')] = f1_score(((predictions_notthresholded_all_organs>threshold)*1).view(-1), groundtruths_thresholded_all_organs)                # self.average[key.replace('y_predicted_', 'all_outputs_auc_score_')] = roc_auc_score(ground_truth, torch.cat(self.values[key],0).view(-1))
                if len(auc_scores)>0:
                    self.average[f'average_auc'] = sum(auc_scores)/len(auc_scores)

        self.values = collections.defaultdict(list)
    
    def get_average(self):
        self.calculate_average()
        return self.average
    
    def add_predictions(self, y_true, y_predicted, y_filter, suffix):
        y_true[y_true==-2] = 0
        y_true[y_true==-1] = 1
        y_true[y_true==-3] = 0
        self.add_list(f'y_true_' + suffix, torch.tensor(y_true))
        self.add_list(f'y_predicted_' + suffix, torch.tensor(y_predicted))
        self.add_list(f'y_filter_' + suffix, torch.tensor(y_filter))

        