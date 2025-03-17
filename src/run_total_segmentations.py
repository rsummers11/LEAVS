# NIH Clinical Center
# Imaging Biomarkers and Computer-Aided Diagnosis Laboratory
# 2025-03-17
# Ricardo Bigolin Lanfredi

import subprocess
import glob
from pathlib import Path
import argparse

import sys

parser = argparse.ArgumentParser(description="Convert CSV format for organ-specific columns.")
parser.add_argument("--start_index", type=int, help="Path to the input CSV file", default=None)
parser.add_argument("--end_index", type=int, help="Path to the output CSV file", default=None)
parser.add_argument("--split", type=str, help="", choices = ['Va', 'Tr'], default=None)
args = parser.parse_args()

filenames = sorted(list(glob.glob(f"datasets/amos/images{args.split}/*.nii.gz")))

if args.end_index is not None and args.start_index is not None:
    filenames = filenames[args.start_index:args.end_index]
elif args.end_index is not None:
    filenames = filenames[:args.end_index]
elif args.start_index is not None:
    filenames = filenames[args.start_index:]

for filename in filenames:
    subprocess.run(f"TotalSegmentator -i {filename} -o datasets/amos/segmentations{args.split}/{Path(filename).stem.split('.', 1)[0]}/ --device gpu -ta total", shell=True)



