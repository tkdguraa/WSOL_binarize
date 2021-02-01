## 1. WSOL training and evaluation

### Datasets

Our repository enables evaluation and training of various WSOL methods on three benchmarks,
CUB and ImageNet, and OpenImages.

### Methods

Implemetation of the methods are provided by [NAVER wsol evaluation](https://github.com/clovaai/wsolevaluation). For more detail, please refer to it.

### Run train+eval

We support the following architectures, methods, and ways to genete CAM:

* Architectures.
  - `vgg16`
  - `inception_v3`
  - `resnet50`
  
* Methods
  - `cam`
  - `has`
  - `acol`
  - `spg`
  - `adl`
  - `cutmix`

* Ways to generate CAMs(neg : nwc, bin : binarize weight to 0 or 1, vote : ours)
  - `origin`
  - `neg`
  - `bin`
  - `vote`

Below is an example command line for the train+eval script.

```bash
python main.py --dataset_name OpenImages \
               --architecture vgg16 \
               --wsol_method cam \
               --experiment_name OpenImages_vgg16_CAM \
               --pretrained TRUE \
               --num_val_sample_per_class 5 \
               --large_feature_map FALSE \
               --batch_size 32 \
               --epochs 10 \
               --lr 0.00227913316 \
               --lr_decay_frequency 3 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 4 \
               --box_v2_metric True \
               --iou_threshold_list 30 50 70 \
               --eval_checkpoint_type last
```

Below is an example command line for the eval script.
For CUB and ILSVRC, this script evaluate the models using follwing metrics.
For fair top1 accuracy, set box_v2_metric as FALSE.
Choose the way to generate CAM by setting mode(origin, neg, bin, vote).
If you use our method _vote_, it needs to set binary_rate.

* metrics.
  - `top1 accuracy`
  - `top1 classification`
  - `gt-known accuracy`

```bash
python main_top1.py --dataset_name CUB \
               --architecture vgg16 \
               --wsol_method cam \
               --experiment_name cub_vgg16_CAM \
               --pretrained TRUE \
               --num_val_sample_per_class 5 \
               --large_feature_map FALSE \
               --batch_size 32 \
               --epochs 10 \
               --lr 0.00227913316 \
               --lr_decay_frequency 3 \
               --weight_decay 5.00E-04 \
               --override_cache FALSE \
               --workers 4 \
               --box_v2_metric FALSE \
               --iou_threshold_list 30 50 70 \
               --mode vote \
               --binary_rate 0.15 \
               --eval_checkpoint_type last
```

See [config.py](config.py) or [config_1.py](config_1.py) for the full descriptions of the arguments, especially the method-specific hyperparameters.
