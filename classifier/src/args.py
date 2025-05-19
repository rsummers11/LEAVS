# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import argparse
import os
import time
import numpy as np
import socket
from random import randint

#convert a few possibilities of ways of inputing boolean values to a python boolean
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_slurm_job_id():
    job_id = os.environ.get('SLURM_JOB_ID')
    return job_id
    
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--amos_folder", default='../datasets/amos/', type=str, help="Folder where the AMOS dataset was downloaded to.")
    parser.add_argument("--labels_file_train_amos", default='../results/qwen2_leavs/parsing_results_llm.csv', type=str, help="Path to the CSV containing the LEAVS outputs for the training set of AMOS")
    parser.add_argument("--labels_file_test_amos", default='../majority_vote_amos_test_annotations.csv', type=str, help="Path to the csv file containing the human annotations on the AMOS dataset, after aggregation by case.")
    
    parser.add_argument('--split_val', type=str, nargs='?', default='val', choices = ['val', 'test'], help='set it to val for using the validation set, and test to use the test set')
    parser.add_argument('--dropout_p', type=float, nargs='?', default=0.9, help='The dropout probability to use between the two resnet layers.')
    
    parser.add_argument('--hidden_channels_conv', type = int, default = 256, help = 'number of channels between the two resnet layers.')
    
    parser.add_argument('--experiment', type=str, required=True,
        help='Set the name of the folder where to save the logs, checkpoints, tensorboard and images for this run.')

    parser.add_argument('--gpus', type=str, default=None,
        help='Set the gpus to use, using CUDA_VISIBLE_DEVICES syntax.')
    
    parser.add_argument("--log_dir", type=str, default="logs/",
        help="The output directory where the model predictions will be written.",)
    
    parser.add_argument("--model_dir", type=str, default="checkpoints/",
        help="The output directory where the model checkpoints will be written.",)

    parser.add_argument("--train_batch_size", type=int, default=512,
        help="Batch size (per device) for the training dataloader.")
    
    parser.add_argument("--eval_batch_size", type=int, default=1,
        help="Batch size (per device) for the validation dataloader.")
    
    parser.add_argument("--num_train_epochs", type=int, default=100, help='how many epochs to train the model for')

    parser.add_argument("--checkpointing_epochs", type=int, default=50,
        help=("Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."),)
    
    parser.add_argument("--checkpoints_total_limit", type=int, default=1,
        help=("Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more details"),)
    
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
        help=("Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'),)
    
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    
    parser.add_argument("--learning_rate", type=float, default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",)
    
    parser.add_argument("--scale_lr", type=str2bool, default="false",
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",)
    
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
        help=("Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."),)
    
    parser.add_argument("--adam_beta1", type=float, default=0.9, 
        help="The beta1 parameter for the Adam optimizer.")
    
    parser.add_argument("--adam_beta2", type=float, default=0.999, 
        help="The beta2 parameter for the Adam optimizer.")
    
    parser.add_argument("--adam_weight_decay", type=float, default=0, 
        help="Weight decay to use.")
    
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, 
        help="Epsilon value for the Adam optimizer")
    
    parser.add_argument("--max_grad_norm", default=None, type=float, 
        help="Max gradient norm. It is not applied if it is not set. Default: None (not applied)")
    
    parser.add_argument("--allow_tf32", type=str2bool, default="false",
        help=("Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"),)
    
    parser.add_argument("--local_rank", type=int, default=-1, 
        help="For distributed training: local_rank")
    
    parser.add_argument("--set_grads_to_none", type=str2bool, default="false",
        help=("Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"),)
    
    parser.add_argument("--validation_metrics_epochs", type=int, default=5,
        help=("Run validation every X epochs."),)

    parser.add_argument('--skip_train', type=str2bool, nargs='?', default='false',
        help='If you just want to run validation, set this value to true.')
    
    parser.add_argument('--skip_validation', type=str2bool, nargs='?', default='false',
        help='If True, no validation is performed.')

    parser.add_argument('--device', type=str, default='cuda',
        help='device string for pytorch')
    parser.add_argument("--min_new_epochs", type=int, default=0, help="a minimmum of epochs to run, regardless of num_train_epochs. Useful when loading a checkpoint")     
    parser.add_argument("--scratch_dir",  type=str, default='./scratch/', help="folder where to put precomputed datasets")

    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    args.ssl_model_to_use = 'uaes'
    args.dataset = ['amos']
    args.dataset_val = 'amos'
    args.segmentation_interpolation_mode = 'nearest'
    args.segmentation_interpolation_antialiasing = False
    args.concat_type_classifier = 'up'
    args.join_levels_classifier = 'cat'
    args.normalize_coarse = True
    args.upsampling_classifier = 'linear'
    args.upsampling_dataset = 'linear'
    args.normal_abnormal = None
    args.do_pos_weight = False
    args.urgency_min = 0
    args.total_iterations = None
    args.minimum_value_foregound_segmentation = 1

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.gpus is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        args.gpus = os.environ['CUDA_VISIBLE_DEVICES']
    
    #gets the current time of the run, and adds a four digit number for getting
    #different folder name for experiments run at the exact same time.
    args.timestamp = time.strftime("%Y%m%d-%H%M%S")
    args.folder_name = '-'.join([args.experiment, args.timestamp, str(randint(10000,99999))])
    
    args.function_to_compare_validation_metric = lambda x,y:x>y #the higher the metric the better
    args.initialization_comparison = float('-inf')

    #register a few values that might be important for reproducibility
    args.screen_name = os.getenv('STY')
    args.hostname = socket.gethostname()
    import platform
    args.python_version = platform.python_version()
    import torch
    args.pytorch_version = torch.__version__ 
    import torchvision
    args.torchvision_version = torchvision.__version__
    args.numpy_version = np.__version__
    args.jobid = get_slurm_job_id()

    return args