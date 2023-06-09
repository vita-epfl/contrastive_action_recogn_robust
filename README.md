# Pose-based Action Recognition: Robust Models with contrastive loss

Semester project

Student: Aleksandra Novikova, Data Science

Supervisor: Mohamed Ossama Ahmed Abdelfattah


## Navigating the repo

- `Report.pdf` - project report

- `TCL/` - implementation of the contrastive loss method with noisy data for training robust models.
The main code is taken from the repository [TCL](https://github.com/CVIR/TCL).
Models (backbones) are taken from repositories [MotionBERT](https://github.com/Walter0807/MotionBERT) and [pySKL](https://github.com/kennymckormick/pyskl).

- `pyskl/` - implementation of different input types for a `PoseConv3D` model.
The main code is taken from the repository [pySKL](https://github.com/kennymckormick/pyskl).

Details can be found in the respective repositories.

## Training 

### Contrastive learning
- PoseConv3D model training:
```
python main.py ntu60 RGB --seed 123 --input_f Skeleton --in_channels 68 --strategy classwise --arch resnet18 --num_segments 8 --second_segments 8 --threshold 0.8 --gd 20  --epochs 1000 --percentage 0.95 -j 2 --dropout 0.5 --consensus_type=avg --eval-freq=1 --print-freq 50 --shift --shift_div=8 --shift_place=blockres --npb --gpus 0  --mu 1 --gamma 5 --gamma2 0 --use_group_contrastive --sup_thresh 0 --batch-size 32 --valbatchsize 32  --lr_backbone1 0.001 --lr_decay1 0.9  --noise_alpha 0.1
```

- MotionBERT model training:
```
python main.py ntu60 RGB --seed 123 --input_f Skeleton --in_channels 68 --strategy classwise --arch resnet18 --num_segments 8 --second_segments 8 --threshold 0.8 --gd 20  --epochs 1000 --percentage 0.95 -j 2 --dropout 0.5 --consensus_type=avg --eval-freq=1 --print-freq 50 --shift --shift_div=8 --shift_place=blockres --npb --gpus 0  --mu 1 --gamma 5 --gamma2 0 --use_group_contrastive --sup_thresh 0 --batch-size 32 --valbatchsize 32  --lr_backbone1 0.001 --lr_decay1 0.9  --noise_alpha 0.1 --model_type motionbert
```

where 
  -  `noise_alpha` - percentage of added noise to data
  -  `gamma` - coefficient before instance contrastive loss
  -  `gamma2` - coefficient before group contrastive loss


### PoseConv3D with different input types

```
bash tools/dist_train.sh configs/posec3d/slowonly_r50_ntu60_xsub/joint.py 1 --validate --test-last --test-best
```

- `configs/posec3d/slowonly_r50_ntu60_xsub/joint.py` for `3D Heatmaps`
- `configs/posec3d/slowonly_r50_ntu60_xsub/joint_grayscale.py` for `Grayscale`
- `configs/posec3d/slowonly_r50_ntu60_xsub/joint_skeleton.py` for `Skeleton`
