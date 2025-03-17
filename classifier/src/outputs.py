# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

# class that manages the saving of logs to the output folder.
# logs are in the form of a txt file, a csv file, a tnsorboard log.
# this class also saves the current source code from the src folder,
# and the configurations used to run the specific training script
import logging
import os
import glob
import shutil
import sys
import csv
from PIL import Image
import numpy as np
import nibabel as nib
import torch
import pandas as pd

def save_image(filepath, numpy_array):
    im = Image.fromarray(((numpy_array*0.5 + 0.5)*255))

    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save(filepath)

class Outputs():
    def __init__(self, opt):
        self.nrows_fixed = 2
        all_folders = [opt.log_dir, opt.model_dir]
        for folder in all_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)
        self.log_dir = os.path.join(opt.log_dir, opt.folder_name)
        self.model_dir = os.path.join(opt.model_dir, opt.folder_name)
        all_folders = [self.log_dir, self.model_dir]
        for folder in all_folders:
            if not os.path.exists(folder):
                os.mkdir(folder)
        for handler in logging.root.handlers:
            logging.root.removeHandler(handler)
        logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                            datefmt="%m/%d/%Y %H:%M:%S",
                            filename = os.path.join(self.log_dir, 'log.txt') ,
                            level = logging.INFO)
        self.csv_file =  os.path.join(self.log_dir, 'log.csv') 
        with open(self.csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['key','value','epoch'])
        self.log_configs(opt)
    
    # Write all the configurations from the opt variable to the log.txt file
    def log_configs(self, opt):
        logging.info('-------------------------------used configs-------------------------------')
        for key, value in sorted(vars(opt).items()):
            logging.info(f"{key}: {value}")
            with open(self.csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([key, value, str(-1)])
        logging.info('-----------------------------end used configs-----------------------------')

    # activate the average calculation for all metric values and save them to log and tensorboard
    def log_added_values(self, epoch, metrics):
        if len(metrics.values['dice_score_'])>0:
            csv_area_file =  os.path.join(self.log_dir, f'area_dice-{epoch}.csv') 
            data = {'Dice': metrics.values['dice_score_'], 'Area': metrics.values['region_area_']}
            df = pd.DataFrame(data)
            df.to_csv(csv_area_file, index=False)
        averages = metrics.get_average()
        logging.info('Metrics for epoch: ' + str(epoch))
        for key, average in averages.items():
            logging.info(f"{key}: {average}")
            with open(self.csv_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([key, str(average), str(epoch)])
        return averages
    
    #save the source files used to run this experiment
    def save_run_state(self, py_folder_to_save):
        if not os.path.exists('{:}/src/'.format(self.log_dir)):
            os.mkdir('{:}/src/'.format(self.log_dir))
        [shutil.copy(filename, ('{:}/src/').format(self.log_dir)) for filename in glob.glob(os.path.join(py_folder_to_save, '*.py'))]
        self.save_command()
    
    #saves the command line command used to run these experiments
    def save_command(self, command = None):
        if command is None:
            command = ' '.join(sys.argv)
        with open("{:}/command.txt".format(self.log_dir), "w") as text_file:
            text_file.write(command)
    
    def save_model_outputs(self, annot, pred, filter, name):
        df = pd.DataFrame({
            'annot': annot,
            'pred': pred,
            'filter': filter
        })

        csv_file_path = f'{self.log_dir}/{name}_model_outputs.csv' 
        
        df.to_csv(csv_file_path, index=False)  

    def save_image(self, image, title, epoch, transpose, spacings, original_shape = None, destination_shape = None):
        if transpose:
            image = image.permute(0, 1, 4, 3, 2)
            if original_shape is not None:
                original_shape = [original_shape[2],original_shape[1],original_shape[0]]
                destination_shape = [destination_shape[2],destination_shape[1],destination_shape[0]]
        image = image[:,0]
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        print(image.shape)
        if len(image.shape)==4:
            path = f'{self.log_dir}/{title}-{epoch}.nii'
            image_stacked = np.hstack(image)
            image_stacked = (image_stacked-image_stacked.min())
            if image_stacked.max()<=1:
                image_stacked = 1024*image_stacked
            orientation = np.eye(4)
            orientation[0][0] = -orientation[0][0]
            orientation[1][1] = -orientation[1][1]
            if destination_shape is not None:
                for i in range(3):
                    orientation[i][i] = orientation[i][i]*(1/destination_shape[i]*original_shape[i]*spacings[i])
            else:
                for i in range(3):
                    orientation[i][i] = orientation[i][i]*(spacings[i])
            img = nib.Nifti1Image((image_stacked).astype(np.int16), orientation)
            nib.save(img, path) 
        if len(image.shape)<4:
            if len(image.shape) == 3:
                image = np.vstack(np.hsplit(np.hstack(image), self.nrows_fixed))
            image = np.rot90(image, k=1, axes=(0, 1))
            path = f'{self.log_dir}/{title}-{epoch}.png'
            save_image(path, image)