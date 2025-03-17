# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import torch
import numpy as np
import torch.nn.functional as F
import os
from types import SimpleNamespace
import math
from tqdm.auto import tqdm
import torch.distributed as dist

def rename_keys_with_substring(state_dict, old_substring, new_substring):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace(old_substring, new_substring)
        new_state_dict[new_key] = value
    return new_state_dict

class ToDouble(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.double()

class ToFloat(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.float()

from functools import wraps
def collect(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return list(func(*args, **kwargs))
    return wrapper

@collect
def yield_sw_boxes(input_size, patch_size, overlap):
    stride = patch_size - overlap
    for start in np.ndindex(tuple(1 + np.int64(np.ceil((input_size - patch_size) / stride)))):
        start *= stride
        stop = np.minimum(start + patch_size, input_size)
        start = stop - patch_size
        yield np.array([start, stop])

def validation_loop(args, main_model, val_dataloader, metric, output, global_step, epoch, pos_weigths):
    if not args.skip_validation: 
        metric.start_time('validation')
        with torch.no_grad():
            main_model.eval()
            if (((epoch+1) % args.validation_metrics_epochs)==0 or epoch==(args.num_train_epochs-1)):

                progress_bar = tqdm(range(0, len(val_dataloader)))
                progress_bar.set_description("Validation loss")
                for step, batch in enumerate(val_dataloader):
                    batch['image'] = batch['image'].to(args.device)
                    batch['labels'] = batch['labels'].to(args.device)
                    batch['urgency'] = batch['urgency'].to(args.device)
                    dec = main_model(batch)
                    loss = loss_ce(args, pos_weigths, do_multiplier = False)(dec,batch['labels'],batch['urgency'])
                    metric.add_value('loss_validation', loss)
                    progress_bar.update(1)
                    # break #TEMP
                progress_bar = tqdm(range(0, len(val_dataloader)))
                progress_bar.set_description("Validation metrics")
                for step, batch in enumerate(val_dataloader):
                    batch['image'] = batch['image'].to(args.device)
                    output_tensor = main_model(batch).cpu()
                    metric.add_predictions(batch['labels'].to(args.device), output_tensor, ((batch['urgency'].to(args.device)>=args.urgency_min) | (batch['urgency'].to(args.device)<0))*1, '') 
                    progress_bar.update(1)
            
                for index_organ in range(batch['labels'].shape[1]):
                    for index_finding in range(batch['labels'].shape[2]):
                        output.save_model_outputs(
                            (torch.cat(metric.values['y_true_'],0)[:,index_organ,index_finding]).numpy(), 
                            (torch.cat(metric.values['y_predicted_'],0)[:,index_organ,index_finding]).numpy(), 
                            (torch.cat(metric.values['y_filter_'],0)[:,index_organ,index_finding]).numpy(), 
                            f'outputs_{index_organ}_{index_finding}')
            main_model.train()
        metric.end_time('validation')

def sum_parameters(model):
    total_sum = 0
    for param in model.parameters():
        total_sum += param.sum().item()
    return total_sum

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

class loss_ce(object):
    def __init__(self, args, pos_weigths, do_multiplier = False):
        self.criterion_bce = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weigths, reduction='none')
        self.uncertainty_label = 1
        self.normal_abnormal = args.normal_abnormal
        self.urgency_min = args.urgency_min
        self.do_multiplier = do_multiplier

    def __call__(self, out, labels, urgency):
        total_loss = []
        unchanged_uncertainties = (labels==-3)* 1
        if self.urgency_min>0:
            unchanged_uncertainties = torch.logical_or(unchanged_uncertainties,  (urgency<=self.urgency_min)*1)*1
        
        out_labels = out
        labels_to_use = labels
        labels_to_use = labels_to_use
        labels_to_use[labels_to_use==-3] = self.uncertainty_label
        labels_to_use[labels_to_use==-2] = 0
        labels_to_use[labels_to_use==-1] = self.uncertainty_label
        
        indice_classification_present = (1-unchanged_uncertainties).bool()
        criterion = self.criterion_bce
        smoothed_labels = labels_to_use
        classification_loss = criterion(out_labels*indice_classification_present, (smoothed_labels*indice_classification_present).float())
        classification_loss = classification_loss.mean()

        total_loss.append(classification_loss)
        return sum(total_loss)/len(total_loss)
    
def train_loop(args, main_model, optimizer, metric, output, train_dataloader, progress_bar, first_epoch, epoch, resume_step, global_step, scaler, pos_weigths):
    step = 0
    if not args.skip_train:
        metric.start_time('train')
        main_model.train()
                
        for index_batch, batch in enumerate(train_dataloader):
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                step+=1
                continue
            batch['image'] = batch['image'].to(args.device)
            
            batch['labels'] = batch['labels'].to(args.device)
            batch['urgency'] = batch['urgency'].to(args.device)
            dec = main_model(batch)
            
            loss = loss_ce(args, pos_weigths, do_multiplier = False)(dec,batch['labels'], batch['urgency'])

            if loss!=loss:
                print(dec)
                1/0
            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(main_model.parameters(), args.max_grad_norm)
            optimizer.step()
                        
            progress_bar.update(1)                
                
            logs = {"loss": loss.detach().item()}

            metric.add_value('loss_training', loss)
            progress_bar.set_postfix(**logs)
            if global_step >= args.max_train_steps:
                break
            global_step += 1
            step += 1
            torch.cuda.empty_cache()
            metric.end_time('train')
    return global_step, step

def do_epochs(args, main_model, optimizer, metric, output, train_dataloader, val_dataloader, progress_bar, first_epoch, resume_step, global_step, scaler, last_best_validation_metric, pos_weigths):
    for epoch in range(first_epoch, max(args.num_train_epochs, first_epoch + args.min_new_epochs)):
        metric.start_time('epoch')
        global_step, step = train_loop(args, main_model, optimizer, metric, output, train_dataloader, progress_bar, first_epoch, epoch, resume_step, global_step, scaler, pos_weigths)
        def checkpoint_save(checkpoint_name):
            checkpoint = {
                "model": main_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "resume_step":0,
                "first_epoch": epoch+1,
                'last_best_validation_metric': last_best_validation_metric,
            }
            save_path = os.path.join(args.model_dir, args.folder_name, f"{checkpoint_name}.temp.new") 
            save_on_master(checkpoint, save_path)
            if os.path.exists(os.path.join(args.model_dir, args.folder_name, checkpoint_name)):
                os.rename(os.path.join(args.model_dir, args.folder_name, checkpoint_name),
                        os.path.join(args.model_dir, args.folder_name, f"{checkpoint_name}.temp.old"))
            os.rename(os.path.join(args.model_dir, args.folder_name, f"{checkpoint_name}.temp.new"),
                    os.path.join(args.model_dir, args.folder_name, checkpoint_name))
            if os.path.exists(os.path.join(args.model_dir, args.folder_name, f"{checkpoint_name}.temp.old")):
                os.remove(os.path.join(args.model_dir, args.folder_name, f"{checkpoint_name}.temp.old"))
        if ((epoch+1)%args.checkpointing_epochs == 0) or epoch == args.num_train_epochs - 1:
            
            checkpoint_save('checkpoint-latest')
            
        validation_loop(args, main_model, val_dataloader, metric, output, global_step, epoch, pos_weigths)
                  
        averages = output.log_added_values(epoch, metric)

        if 'average_auc' in averages and args.function_to_compare_validation_metric(averages['average_auc'],last_best_validation_metric):
            checkpoint_save('checkpoint-best')
            last_best_validation_metric = averages['average_auc']  

        metric.end_time('epoch')
        metric.start_time('full_script')

from datasets import custom_collate_train
from torch.utils.data._utils.collate import default_collate
def newcf(batch):
    item1, item2 = batch[0]
    item2 = custom_collate_train([item2])
    item1 = default_collate([item1])
    item1.update({key+'2': value for key, value in item2.items()})
    return item1

class ConcatenatedDataset:
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, idx):
        item1 = self.dataset1[idx]
        item2 = self.dataset2[idx]
        return item1, item2

    def __len__(self):
        return len(self.dataset1)
    
def concatenatedDataLoader(loader1, loader2, split):
    assert len(loader1) == len(loader2), f"DataLoaders must have the same length, {len(loader1)} and {len(loader2)}"
    assert loader1.batch_size == loader2.batch_size, "DataLoaders must have the same batch size"
    train_dataloader = torch.utils.data.DataLoader(
        ConcatenatedDataset(loader1.dataset,loader2.dataset),
        batch_size=loader1.batch_size,
        shuffle=(split=='train'),
        num_workers=loader1.num_workers,
        pin_memory=loader1.pin_memory,
        drop_last = split=='train',
        collate_fn = newcf
    )
    return train_dataloader

def replace_keys(state_dict, old_substring='.model.', new_substring='.module.'):
    new_state_dict = {key.replace(old_substring, new_substring): value for key, value in state_dict.items()}
    return new_state_dict   
               
def main(args):
    args.dataset_directory = args.amos_folder    
    datamodules = []
    datamodules_eval = []
    if args.ssl_model_to_use == 'uaes':
        import sys
        sys.path.append('./sam/tools')
        import interfaces
        config_file = 'sam/configs/samv2/samv2_NIHLN.py'
        checkpoint_file = 'sam/checkpoints/SAMv2_iter_20000.pth'
        embedding_model = interfaces.init(config_file, checkpoint_file, cpu = args.device=='cpu')

        def embedding_inference(input_patch, embedding_model_):
            embedding_model_.eval()
            assert(len(input_patch['img_metas'])==1)
            
            
            embedded = embedding_model_(return_loss=False, rescale=True, **(input_patch['img_metas'][0]))
            
            return embedded[0:2]
    
    for ssl_model_to_use in ([args.ssl_model_to_use]):
        if ssl_model_to_use == 'uaes':
            
            from datasets import get_dataset_sam as get_loader
            embedder_arguments = {}
            embedder_arguments["space_x"] = 2.0
            embedder_arguments["space_y"] = 2.0
            embedder_arguments["space_z"] = 2.0
            args.space_x = embedder_arguments["space_x"]
            args.space_y = embedder_arguments["space_y"]
            args.space_z = embedder_arguments["space_z"]
            embedder_arguments["dataset_directory"] = args.dataset_directory
            embedder_arguments["upsampling_dataset"] = args.upsampling_dataset
            embedder_arguments["labels_file_train_amos"] = args.labels_file_train_amos
            embedder_arguments["labels_file_test_amos"] = args.labels_file_test_amos
            
            embedder_arguments["dataset"] = args.dataset
            embedder_arguments["normal_abnormal"] = args.normal_abnormal
            embedder_arguments["segmentation_interpolation_mode"] = args.segmentation_interpolation_mode
            embedder_arguments["segmentation_interpolation_antialiasing"] = args.segmentation_interpolation_antialiasing
            embedder_arguments["minimum_value_foregound_segmentation"] = args.minimum_value_foregound_segmentation
            embedder_arguments["embedding_size"] = 128*3
            embedder_arguments = SimpleNamespace(**embedder_arguments)
            args.embedding_size = embedder_arguments.embedding_size
            args.embedding_levels = 2

        from datasets import get_dataloader
        val_dataset, val_post_transform, _ = get_loader(embedder_arguments,args.split_val, embedding_inference, embedding_model, args.dataset_val)
        test_loader = get_dataloader(args, [args.dataset_val], args.split_val, [val_dataset], [val_post_transform], embedder = ssl_model_to_use)
        train_dataset_list = []
        train_post_transform_list = []
        total_matrices = 0
        positive_matrices = 0
        for dataset in args.dataset:
            train_dataset, train_post_transform, count_matrices = get_loader(embedder_arguments,'train', embedding_inference, embedding_model, dataset)
            train_dataset_list.append(train_dataset)
            train_post_transform_list.append(train_post_transform)
            total_matrices = total_matrices + count_matrices['total']
            print(count_matrices)
            positive_matrices = positive_matrices + count_matrices['positives']
        if args.do_pos_weight:
            pos_weigths = (total_matrices - positive_matrices)/positive_matrices
            pos_weigths[np.isinf(pos_weigths)] = 1
            pos_weigths = torch.tensor(pos_weigths).to(args.device)
            pos_weigths = torch.clamp(pos_weigths, min=0.1, max=10)
        else:
            pos_weigths = None
        train_loader = get_dataloader(args, args.dataset, 'train', train_dataset_list, train_post_transform_list, embedder = ssl_model_to_use)
        datamodules.append(train_loader)
        datamodules_eval.append(test_loader)
        del embedder_arguments
    if len(datamodules)>1:
        train_loader = concatenatedDataLoader(datamodules[0], datamodules[1], 'train')
        test_loader = concatenatedDataLoader(datamodules_eval[0], datamodules_eval[1], args.split_val)
    else:
        train_loader = datamodules[0]
        test_loader = datamodules_eval[0]
    
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    import outputs
    import metrics
    output = outputs.Outputs(args)
    output.save_run_state(os.path.dirname(__file__))
    metric = metrics.Metrics(args)
    metric.start_time('full_script')
    main_model = None

    from classifier import get_classifier
    if main_model is None:
        classifier, upconv, skip_blocks, JoinModels = get_classifier(args, embedding_inference)

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.total_iterations if args.total_iterations is not None else args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.scale_lr:
        args.learning_rate = (args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size)

    if main_model is None:
        main_model = JoinModels(embedding_model,classifier, upconv, skip_blocks).to(args.device)
    main_model = main_model.to(args.device)
    optimizer = torch.optim.AdamW(
        list(classifier.parameters()) + list(upconv.parameters()) + list(skip_blocks.parameters()),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    scaler = None
    if args.resume_from_checkpoint:
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        def rename_keys(model_state_dict, key_map):
            new_state_dict = {}
            for key in model_state_dict:
                if key in key_map:
                    new_key = key_map[key]
                    new_state_dict[new_key] = model_state_dict[key]
                else:
                    new_state_dict[key] = model_state_dict[key]
            return new_state_dict
        key_map = {"classifier.0.weight":"classifier.0.0.weight", 
         "classifier.0.bias": "classifier.0.0.bias", 
         "classifier.2.weight":"classifier.0.2.weight", 
         "classifier.2.bias": "classifier.0.2.bias"}
        
        main_model.load_state_dict(rename_keys(checkpoint["model"], key_map))

        global_step = checkpoint["global_step"]
        first_epoch = checkpoint["first_epoch"]
        resume_step = checkpoint["resume_step"]
        last_best_validation_metric = checkpoint["last_best_validation_metric"]
        if not args.skip_train:
            optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        global_step = 0
        first_epoch = 0
        resume_step = 0
        last_best_validation_metric = args.initialization_comparison

    progress_bar = tqdm(range(global_step, args.max_train_steps))
    progress_bar.set_description("Steps")

    do_epochs(args, main_model, optimizer, metric, output, train_loader, test_loader, progress_bar, first_epoch, resume_step, global_step, scaler, last_best_validation_metric, pos_weigths)

if __name__ == "__main__":
    from args import parse_args
    args = parse_args()
    main(args)