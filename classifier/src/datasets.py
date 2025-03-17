# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import random
import os
import torch
import sys
import torchvision
import numpy as np
import torch.nn.functional as F
import math
import skimage
import skimage.transform
from scipy import ndimage as ndi
import random
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
from torch.utils.data import ConcatDataset

total_v1 =  {
            1: "spleen",
            2: "kidney_right",
            3: "kidney_left",
            4: "gallbladder",
            5: "liver",
            6: "stomach",
            7: "aorta",
            8: "inferior_vena_cava",
            9: "portal_vein_and_splenic_vein",
            10: "pancreas",
            11: "adrenal_gland_right",
            12: "adrenal_gland_left",
            13: "lung_upper_lobe_left",
            14: "lung_lower_lobe_left",
            15: "lung_upper_lobe_right",
            16: "lung_middle_lobe_right",
            17: "lung_lower_lobe_right",
            18: "vertebrae_L5",
            19: "vertebrae_L4",
            20: "vertebrae_L3",
            21: "vertebrae_L2",
            22: "vertebrae_L1",
            23: "vertebrae_T12",
            24: "vertebrae_T11",
            25: "vertebrae_T10",
            26: "vertebrae_T9",
            27: "vertebrae_T8",
            28: "vertebrae_T7",
            29: "vertebrae_T6",
            30: "vertebrae_T5",
            31: "vertebrae_T4",
            32: "vertebrae_T3",
            33: "vertebrae_T2",
            34: "vertebrae_T1",
            35: "vertebrae_C7",
            36: "vertebrae_C6",
            37: "vertebrae_C5",
            38: "vertebrae_C4",
            39: "vertebrae_C3",
            40: "vertebrae_C2",
            41: "vertebrae_C1",
            42: "esophagus",
            43: "trachea",
            44: "heart_myocardium",
            45: "heart_atrium_left",
            46: "heart_ventricle_left",
            47: "heart_atrium_right",
            48: "heart_ventricle_right",
            49: "pulmonary_artery",
            50: "brain",
            51: "iliac_artery_left",
            52: "iliac_artery_right",
            53: "iliac_vena_left",
            54: "iliac_vena_right",
            55: "small_bowel",
            56: "duodenum",
            57: "colon",
            58: "rib_left_1",
            59: "rib_left_2",
            60: "rib_left_3",
            61: "rib_left_4",
            62: "rib_left_5",
            63: "rib_left_6",
            64: "rib_left_7",
            65: "rib_left_8",
            66: "rib_left_9",
            67: "rib_left_10",
            68: "rib_left_11",
            69: "rib_left_12",
            70: "rib_right_1",
            71: "rib_right_2",
            72: "rib_right_3",
            73: "rib_right_4",
            74: "rib_right_5",
            75: "rib_right_6",
            76: "rib_right_7",
            77: "rib_right_8",
            78: "rib_right_9",
            79: "rib_right_10",
            80: "rib_right_11",
            81: "rib_right_12",
            82: "humerus_left",
            83: "humerus_right",
            84: "scapula_left",
            85: "scapula_right",
            86: "clavicula_left",
            87: "clavicula_right",
            88: "femur_left",
            89: "femur_right",
            90: "hip_left",
            91: "hip_right",
            92: "sacrum",
            93: "face",
            94: "gluteus_maximus_left",
            95: "gluteus_maximus_right",
            96: "gluteus_medius_left",
            97: "gluteus_medius_right",
            98: "gluteus_minimus_left",
            99: "gluteus_minimus_right",
            100: "autochthon_left",
            101: "autochthon_right",
            102: "iliopsoas_left",
            103: "iliopsoas_right",
            104: "urinary_bladder"
        }
inverted_total_v1 = {v: k for k, v in total_v1.items()}

def get_files_with_extension(root_directory, extensions):
    matching_files = {}
    matching_files = []
    for root, dirs, files in os.walk(root_directory):
        for file in files:
            for extension in extensions:
                if file.endswith(extension) and not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    matching_files.append(file_path)
    return matching_files

def split_list_with_seed(data_list, ratios, seed=None):
    if sum(ratios) != 1.0:
        raise ValueError("Ratios must sum to 1.0")

    if seed is not None:
        random.seed(seed)

    random.shuffle(data_list)

    parts = []
    start_index = 0

    for ratio in ratios:
        end_index = start_index + int(len(data_list) * ratio)
        parts.append(data_list[start_index:end_index])
        start_index = end_index

    return parts

def max_interpolation(input, output_size):
    if input.dim() == 3:
        assert output_size is not None
        return torch.nn.functional.adaptive_max_pool1d(input, output_size)
    if input.dim() == 4:
        assert output_size is not None
        return torch.nn.functional.adaptive_max_pool2d(input, output_size)
    if input.dim() == 5:
        assert output_size is not None
        return torch.nn.functional.adaptive_max_pool3d(input, output_size)

BODY_THRESHOLD_HU = -500
from skimage.segmentation import flood
from imops import pad
def get_body_mask(image, threshold_body):
    air = image < threshold_body
    to_return = ~flood(pad(air[0][0], padding=1, axis=(0, 1), padding_values=True), seed_point=(0, 0, 0))[1:-1, 1:-1][None][None]
    return to_return

def masked_avg_pool2d(input, mask):
    # Ensure the mask has the same shape as the input
    if mask.shape[-3:] != input.shape[-3:]:
        raise ValueError("Mask and input must have the same shape")
    to_return=[]
    for index_organ in range(mask.shape[0]):
        # Apply the mask to the input
        masked_input = input * mask[index_organ][None]

        # Calculate the sum of the masked input values
        masked_sum = masked_input.sum(dim=(-3, -2, -1))

        # Calculate the sum of the mask values
        mask_sum = mask[index_organ][None].float().sum(dim=(-3, -2, -1))

        # Calculate the masked average pool
        masked_avg = masked_sum / (mask_sum + 1e-10)
        to_return.append(masked_avg)
    return torch.cat(to_return, 0)

def maximum_avg_pool2d(input, mask):
    # Ensure the mask has the same shape as the input
    if mask.shape[-3:] != input.shape[-3:]:
        raise ValueError("Mask and input must have the same shape")
    to_return=[]
    for index_organ in range(mask.shape[0]):
        # Replace masked values with a very low value (assuming input is positive)
        masked_tensor = torch.where(mask[index_organ][None].bool(), input, torch.tensor(float('-inf'),  dtype=input.dtype,device=input.device))
        
        # Apply max pooling
        pooled = torch.amax(masked_tensor, dim=(2, 3, 4))
        if (mask[index_organ]==0).all():
            pooled = torch.zeros_like(pooled)
        to_return.append(pooled)
    return torch.cat(to_return, 0)

def minimum_avg_pool2d(input, mask):
    return -maximum_avg_pool2d(-input, mask)

class SAMPostTransform():
    def __init__(self,args):
        self.normal_abnormal = args.normal_abnormal
        
    def __call__(self, x):
        x['image'] = torch.tensor(x['image']).float()
        
        if self.normal_abnormal is not None:
            x['labels'] = convert_array_to_ordered(x['labels'])
            self.normal_abnormal = list(map(int, self.normal_abnormal))
            smoothed_labels = 0
            smoothed_urgency = 0
            for index_abnormality_type in self.normal_abnormal:
                smoothed_labels = np.maximum(smoothed_labels, x['labels'][:,index_abnormality_type][...,None])
                smoothed_urgency = np.maximum(smoothed_urgency, x['urgency'][:,index_abnormality_type][...,None])
            x['urgency'] = smoothed_urgency
            x['labels'] = convert_array_to_unordered(smoothed_labels)
        
        x['labels'] = torch.tensor(x['labels']).float()
        x['urgency'] = torch.tensor(x['urgency']).float()
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

def convert_array_to_ordered(array_numbers):
    mapping = {1: 4, -1: 3, -3: 2, -2: 1, 0: 0}
    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    return vectorized_mapping(array_numbers)

def convert_array_to_unordered(array_numbers):
    mapping = {4:1, 3: -1, 2: -3, 1: -2, 0:0}
    vectorized_mapping = np.vectorize(lambda x: mapping[x])
    return vectorized_mapping(array_numbers)

def convert_list_to_ordered(list_numbers):
    # Define the mapping
    mapping = {1: 4, -1: 3, -3: 2, -2: 1, 0:0}
    # Use list comprehension for concise mapping
    return [mapping[annot] for annot in list_numbers]
    
def convert_list_to_unordered(list_numbers):
    # Define the mapping
    mapping = {4:1, 3: -1, 2: -3, 1: -2, 0:0}
    # Use list comprehension for concise mapping
    return mapping[list_numbers]

def get_dataset_sam(args, split, embedding_inference, embedding_model, dataset):
    sys.path.append('./sam/tools')
    import utils as utils_dataset_sam

    labels_tables = None
    if dataset=='amos':
        if split =='test':
            df = pd.read_csv(args.labels_file_test_amos)
            df = df.drop(columns=["index"], errors="ignore") 
            # Identify general columns and organ-specific prefixes
            general_cols = ['image1', 'subjectid_studyid', 'type_annotation']

            # Melt the dataframe to reshape it
            melted_df = df.melt(id_vars=general_cols, var_name='feature', value_name='value')
            # Extract the organ name and feature type
            melted_df[['organ', 'feature_type']] = melted_df['feature'].str.extract(r'^(.*?)_(.*)$')
            melted_df['subjectid_studyid'] = melted_df['subjectid_studyid'].fillna("")
            melted_df['value'] = melted_df['value'].fillna(-1)
            # Pivot the table to have the required structure
            final_df = melted_df.pivot_table(index=general_cols + ['organ'], 
                                            columns='feature_type', 
                                            values='value', 
                                            aggfunc='first').reset_index()
            # Rename columns to match the requested format
            final_df.columns.name = None  # Remove index name
            labels_tables = final_df
        else:
            labels_tables = pd.read_csv(args.labels_file_train_amos)

    if dataset=='amos':
        dataset_directory = args.dataset_directory + ('imagesVa/' if split=='test' else 'imagesTr/')
        
        all_nii_files = get_files_with_extension(dataset_directory, ['.nii.gz'])
        if split!='test':
            ratios = [0.8, 0.2]
            random_seed = 42
            
            split_parts = split_list_with_seed(all_nii_files, ratios, seed=random_seed)[{'val':1,'train':0}[split]]
            val_files =     [
                {'image': element, 'segmentation':element.replace('/images','/segmentations').replace('.nii.gz','/')} for element in split_parts
            ]
        else:
            dataset = "amos_test"
            split_parts = all_nii_files
            val_files =     [
                    {'image': element, 'segmentation':element.replace('/images','/segmentations').replace('.nii.gz','/')} for element in split_parts if Path(element).name+'.txt' in labels_tables['image1'].values
                ]

    #torchio "LPS+" direction which equal to "RAI" orientation in ITK-Snap.

    class SamDataset(torch.utils.data.Dataset):
        def __init__(self, files, dataset, labels_tables, minimum_value_foregound_segmentation, segmentation_interpolation_mode, segmentation_interpolation_antialiasing):
            self.files = files
            self.dataset = dataset
            self.labels_tables = labels_tables
            self.minimum_value_foregound_segmentation = minimum_value_foregound_segmentation
            self.segmentation_interpolation_mode = segmentation_interpolation_mode
            self.segmentation_interpolation_antialiasing = segmentation_interpolation_antialiasing

        def __len__(self):
            return len(self.files)

        def __getitem__(self,index):
            img_orig, img, _ = utils_dataset_sam.read_image(self.files[index]['image'], norm_spacing=(args.space_x, args.space_y, args.space_z)) #the order of xyz might be wrong
            from global_ import organ_denominations, finding_types
            if self.dataset=='amos_test':
                from global_ import finding_types_human as finding_types
            organ_dict = {'small bowel': ['duodenum', 'small_bowel'], 'large bowel': ['colon'],'right kidney':['kidney_right'],'left kidney':['kidney_left']}
            seg =[]
            if self.files[index]['segmentation'][-7:]=='.nii.gz':
                all_segs = utils_dataset_sam.read_image(self.files[index]['segmentation'], reference_img_path = self.files[index]['image'], redimension = False, rescale = False)[1]['img'][0].numpy()

            for organ in organ_denominations:
                seg_ = None

                for organ_string in (organ_dict[organ] if organ in organ_dict else [organ]):
                    if self.files[index]['segmentation'][-7:]=='.nii.gz':
                        seg_to_add = (all_segs==inverted_total_v1[organ_string])*1
                    else:
                        seg_to_add = utils_dataset_sam.read_image(self.files[index]['segmentation'] + organ_string + '.nii.gz', reference_img_path = self.files[index]['image'], redimension = False, rescale = False)[1]['img'][0].numpy()
                    if seg_ is None:
                        seg_ = seg_to_add
                    else:
                        seg_ = seg_ + seg_to_add
                seg.append(seg_)
            seg = np.concatenate(seg, axis = 0)
            img['img_metas'] = img['img_metas'].data
            img['img'][0] = img['img'][0].numpy()
            threshold_body = ((BODY_THRESHOLD_HU+1024)/(3071+1024)*255-50)
        
            body_mask = get_body_mask(img['img'][0], threshold_body) 
            if self.labels_tables is not None:
                labels_tables = self.labels_tables
                if self.dataset=='amos':
                    labels_tables = labels_tables[labels_tables['subjectid_studyid'].str[:27]==self.files[index]['image'].replace(args.dataset_directory, './')]
                if self.dataset=='amos_test':
                    labels_tables = labels_tables[labels_tables['image1']==Path(self.files[index]['image']).name +'.txt']
                elif self.dataset=='deep':
                    id_deep = str(int(str(Path(self.files[index]['image']).parent.name)))
                    labels_tables = labels_tables[labels_tables['subjectid_studyid']==(id_deep + "_" + id_deep)]


            else:
                seg_label = utils_dataset_sam.read_image(self.files[index]['image'].replace('/images/','/labels/'), reference_img_path = self.files[index]['image'], redimension = False, rescale = False)[1]['img'][0].numpy()
                list_rows = []
                for organ in organ_denominations:
                    labels = {finding_type: (((seg_label==2).sum()>0)*1 if ((organ=='liver' if self.dataset=='lits' else organ=='pancreas') and finding_type=='focal') else 0) for finding_type in set(value for lst in finding_types.values() for value in lst)}
                    urgencies = {finding_type: 0 for finding_type in set(value for lst in finding_types.values() for value in lst)}
                    row = {'type_annotation':'labels', 'organ':organ}
                    row.update(labels)
                    list_rows.append(row)
                    row = {'type_annotation':'urgency', 'organ':organ}
                    row.update(urgencies)
                    list_rows.append(row)
                labels_tables = pd.DataFrame(list_rows)
            
            labels = []
            urgency = []
            for organ in organ_denominations:
                labels.append([])
                urgency.append([])
                for finding_type in finding_types:
                    this_organ = labels_tables[labels_tables['organ']==organ]
                    if len(finding_types[finding_type])>0:
                        if len(this_organ[this_organ['type_annotation']=='labels'][finding_types[finding_type]].values)==0:
                            print(organ, finding_type, finding_types[finding_type], self.files[index]['image'])
                            1/0
                        labels[-1].append(convert_list_to_unordered(max(convert_list_to_ordered(this_organ[this_organ['type_annotation']=='labels'][finding_types[finding_type]].values[0].tolist()))))
                        urgency[-1].append((max((this_organ[this_organ['type_annotation']=='urgency'][finding_types[finding_type]].values[0].tolist()))))
                    else:
                        labels[-1].append(0)
                        urgency[-1].append(-1)

            labels = np.stack(labels,axis=0)
            urgency = np.stack(urgency,axis=0)

            x = {'image':img, 'labels':labels, 'segmentation':seg, 'urgency':urgency}

            x['img_metas'] = x['image']
        
            x['img_metas']['img_metas'] = x['img_metas']['img_metas'][0]
            x['img_metas']['img_metas'][0][0]['filename'] = x['img_metas']['img_metas'][0][0]['filename']

            x['image']['img'][0] = torch.tensor(x['image']['img'][0])
            x['image'] = x['image']['img'][0][0]
            
            x["segmentation"] = (x["segmentation"]>=self.minimum_value_foregound_segmentation)*1.
            x["segmentation"] = torch.tensor(x["segmentation"])
            destination_shape = np.array(x["image"].shape[-3:])/2
            destination_shape[0] = math.floor(destination_shape[0])
            destination_shape[1] = math.ceil(destination_shape[1])
            destination_shape[2] = math.ceil(destination_shape[2])
            destination_shape = tuple(destination_shape.astype(int).tolist())
            new_segmentations = []
            for index_channel in range(x["segmentation"].shape[0]):
                if self.segmentation_interpolation_mode=='max':
                    if self.segmentation_interpolation_antialiasing:
                        factors = np.divide(x["segmentation"][index_channel][0].numpy().shape, destination_shape)
                        anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
                        filtered = torch.tensor(ndi.gaussian_filter(
                            x["segmentation"][index_channel][0].numpy().astype(np.float32), anti_aliasing_sigma, cval=0, mode='reflect'
                        )[None][None])
                    else:
                        filtered = x["segmentation"][index_channel][0][None][None]
                    new_segmentation = max_interpolation(filtered, destination_shape)
                elif self.segmentation_interpolation_mode=='nearest':
                    new_segmentation = torch.tensor(skimage.transform.resize(x["segmentation"][index_channel][0].numpy(), destination_shape, order = 0, anti_aliasing = self.segmentation_interpolation_antialiasing)[None][None])
                elif self.segmentation_interpolation_mode=='box' or self.segmentation_interpolation_mode=='area':
                    factors = np.divide(x["segmentation"][index_channel][0].numpy().shape, destination_shape)
                    anti_aliasing_sigma = np.maximum(0, (factors - 1) / 2)
                    if self.segmentation_interpolation_antialiasing:
                        filtered = torch.tensor(ndi.gaussian_filter(
                            x["segmentation"][index_channel][0].numpy().astype(np.float32), anti_aliasing_sigma, cval=0, mode='reflect'
                        )[None][None])
                    else:
                        filtered = x["segmentation"][index_channel][0][None][None]
                    new_segmentation = F.interpolate(filtered, destination_shape, mode=self.segmentation_interpolation_mode)
                else:
                    if self.segmentation_interpolation_mode=='linear':
                        order = 1
                    elif self.segmentation_interpolation_mode=='cubic':
                        order = 3
                    new_segmentation = torch.tensor(skimage.transform.resize(x["segmentation"][0][index_channel].numpy(), destination_shape, order = order, anti_aliasing = self.segmentation_interpolation_antialiasing)[None][None])

                new_segmentations.append(new_segmentation)

            x["segmentation"] = torch.stack(new_segmentations, axis = 2)[0][0]
            embedded = embedding_inference({'img_metas':[x['img_metas']]}, embedding_model)
            if not isinstance(embedded, (list, tuple)):
                embedded = [embedded]
            if 'nearest' in args.upsampling_dataset:
                def up(x, destination_shape):
                    return F.interpolate(x, destination_shape, mode='nearest')
            elif 'linear' in args.upsampling_dataset:
                up = lambda x, destination_shape: F.interpolate(x, destination_shape, mode='trilinear', align_corners=False)
            
            embedded_resized = []
            for i in range(len(embedded)):
                if embedded[i].shape[-3:]!=x['segmentation'].shape[-3:]:
                    embedded_resized.append(up(embedded[i], x['segmentation'].shape[-3:]))
                else:
                    embedded_resized.append(embedded[i])
            embedded = torch.cat(embedded_resized,axis=1).detach()   
            x1 = masked_avg_pool2d(embedded,x['segmentation'][:,None].cuda()).cpu().numpy()
            x2 = maximum_avg_pool2d(embedded,x['segmentation'][:,None].cuda()).cpu().numpy()
            x3 = minimum_avg_pool2d(embedded,x['segmentation'][:,None].cuda()).cpu().numpy()
            assert((np.abs(x2)<1000).all())
            assert((np.abs(x3)<1000).all())

            return {'image':np.concatenate((x1,x2,x3),axis = 1) , 'labels':labels, 'urgency':urgency}
    
    if split=='train':
        from global_ import organ_denominations, finding_types
        # Create a column that maps the organ to an integer index (0-8)
        organ_dict = {organ: idx for idx, organ in enumerate(organ_denominations)}
        labels_tables_only_labels = labels_tables[labels_tables['type_annotation']=='labels']
        labels_tables_only_labels['organ_idx'] = labels_tables_only_labels['organ'].map(organ_dict)

        # Initialize the dictionary to store count matrices for each value (-3 to 1)
        count_matrices = {}
        all_columns = set(col for subtypes in finding_types.values() for col in subtypes)
        finding_columns = [column for column in labels_tables_only_labels.columns if column in all_columns]

        mask = (labels_tables_only_labels[finding_columns] == -1) | (labels_tables_only_labels[finding_columns] == 1)

        # Add the organ_idx column to the mask
        mask['organ_idx'] = labels_tables_only_labels['organ_idx']

        grouped_mask = pd.DataFrame()

        # Iterate through each finding type in the dictionary
        for group, columns in finding_types.items():
            # Apply the OR function across the columns for this group
            grouped_mask[group] = mask[columns].any(axis=1)

        # Keep the columns that are not in finding_types the same
        remaining_columns = [col for col in mask.columns if col not in grouped_mask.columns]
        grouped_mask[remaining_columns] = mask[remaining_columns]
        
        # Reshape the DataFrame into a long format suitable for crosstab
        reshaped_df = grouped_mask.melt(id_vars=['organ_idx'], value_vars=finding_types.keys(), var_name='finding_type')


        count_matrix = pd.crosstab(reshaped_df['organ_idx'], reshaped_df['finding_type'], dropna=False)
        count_matrix = count_matrix.reindex(columns=finding_types.keys(), fill_value=0)
        count_matrices['total'] = count_matrix.to_numpy()
        reshaped_df = reshaped_df[reshaped_df['value']]
        # Count occurrences using crosstab
        count_matrix = pd.crosstab(reshaped_df['organ_idx'], reshaped_df['finding_type'], dropna=False)
        count_matrix = count_matrix.reindex(columns=finding_types.keys(), fill_value=0)
        # Store the count matrix in the dictionary with the value as the key
        count_matrices['positives'] = count_matrix.to_numpy()
    else:
        count_matrices = None
    val_ds = SamDataset(val_files, dataset, labels_tables, args.minimum_value_foregound_segmentation, args.segmentation_interpolation_mode, args.segmentation_interpolation_antialiasing)
    post_val_transforms = [SAMPostTransform(args)]

    post_val_transforms = torchvision.transforms.Compose(post_val_transforms)
    return val_ds, post_val_transforms, count_matrices

def scale_hu(image_hu, window_hu):
    min_hu, max_hu = window_hu
    assert min_hu < max_hu
    return np.clip((image_hu - min_hu) / (max_hu - min_hu), 0, 1)

def find_smallest_crop_so_roi_contains_with_true(arr, roi_size):
    true_indices = np.argwhere(arr)
    min_indices = np.min(true_indices, axis=0)
    max_indices = np.max(true_indices, axis=0)
    
    # Calculate the dimensions of the largest box that contains all true values
    box_dimensions = max_indices - min_indices + 1
    
    # Calculate the padding required to fit the ROI size
    padding = (roi_size - box_dimensions) // 2
    
    # Ensure that the resulting box accommodates the ROI size
    min_indices = np.maximum(0, min_indices - padding)
    max_indices = np.minimum(arr.shape, max_indices + padding)
    
    slices = tuple(slice(min_idx, max_idx + 1) for min_idx, max_idx in zip(min_indices, max_indices))
    return slices

def find_smallest_box_with_true(arr):
    true_indices = np.argwhere(arr)
    min_indices = np.min(true_indices, axis=0)
    max_indices = np.max(true_indices, axis=0)
    slices = tuple(slice(min_idx, max_idx + 1) for min_idx, max_idx in zip(min_indices, max_indices))
    return slices

#dataset wrapper to apply transformations to a pytorch dataset. indices_to_transform defines the indices of the elements
# of the tuple returned by original_dataset to which the transformation should be applied
class TransformsDataset(Dataset):
    def __init__(self, original_dataset, original_transform):
        self.original_transform = original_transform
        self.original_dataset = original_dataset
    
    def __len__(self):
        return len(self.original_dataset)
    
    def __getattr__(self, attr):
        return getattr(self.original_dataset, attr)
        
    def __getitem__(self, index):
        return self.original_transform(self.original_dataset[index])

def custom_collate_train(batch):
    joined_batch = {}
    for key in batch[0].keys():
        joined_batch[key] = [sample[key] for sample in batch]
        if key in ['image','segmentation', 'segmentation_gt', 'labels', 'urgency']:
            joined_batch[key] = torch.stack(joined_batch[key])
    return joined_batch

def custom_collate_val(batch):
    return batch

def get_dataloader(args, datasets, split, dataset_to_use, post_transform, embedder = None):
    if embedder is None:
        embedder = args.ssl_model_to_use
    from h5_dataset import H5ComposeFunction, change_np_type_fn, PackBitArray, UnpackBitArray, H5Dataset
    h5_datasets = []
    for index_dataset, dataset in enumerate(datasets):
        preprocessing_functions = {}
        postprocessing_functions = {} 
        h5_dataset = H5Dataset(path = args.scratch_dir, 
            filename = f"{'sammaxmin7' if embedder in ['uaes', 'uaem'] else embedder}_{split}_{dataset}_{args.upsampling_dataset}", 
            fn_create_dataset = lambda: dataset_to_use[index_dataset], 
            individual_datasets = True,
            preprocessing_functions = preprocessing_functions, 
            postprocessing_functions = postprocessing_functions,
            n_processes = 1, batch_multiplier = 1, load_to_memory = False)
        h5_dataset = TransformsDataset(h5_dataset, post_transform[index_dataset])
        h5_datasets.append(h5_dataset)
    batch_size = args.eval_batch_size if split!='train' else args.train_batch_size
    combined_dataset = ConcatDataset(h5_datasets)
    train_dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=split=='train',
        num_workers=args.dataloader_num_workers,
        pin_memory=False,
        drop_last = split=='train',
        collate_fn = custom_collate_train,
    )

    return train_dataloader

