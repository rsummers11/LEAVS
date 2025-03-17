# LEAVS classifier

To learn about all the arguments, run `python src/train.py --help`.

Example to run training, validation and testing:

```
python src/train.py --experiment=val_1
python src/train.py --experiment=test_1 --split_val test --skip_train true --validation_metrics_epochs 1  --resume_from_checkpoint checkpoints/val_1-<timestamp>/checkpoint-best
```

Install the following library: https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2/tree/main

Place the following files/folders from https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2/tree/main
in the 'sam' folder:
```
'sam/configs/samv2/samv2_NIHLN.py'
'sam/checkpoints/SAMv2_iter_20000.pth'
'./sam/tools'
```

## Requirements

It was tested with

- python                    3.7.16
- sam library from https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2/tree/main
- torch                     1.9.0+cu111
- torchvision               0.10.0+cu111
- torchaudio                0.9.0
- scipy                     1.7.3
- scikit-learn              1.0.2
- scikit-image              0.19.3
- pandas                    1.3.5
- h5py                      3.8.0
- imops                     0.9.1
- dill                      0.3.7
- numpy                     1.21.6
- imageio                   2.19.3
- mmcv-full                 1.7.2 
- mmdet                     2.27.0  
- mmengine                  0.10.3
- opencv-python             4.9.0.80
- torchio                   0.18.92
- nibabel                   4.0.2
- matplotlib                3.1.0
- joblib                    1.1.1

