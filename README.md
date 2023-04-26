# Visual Attention Network (VAN) for Detection

This repo is a PyTorch implementation applying **VAN** (**Visual Attention Network**) to 2D Object Detection.
Our implementation is mainly based on [VAN-Segmentation](https://github.com/Visual-Attention-Network/VAN-Segmentation) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Thanks to the authors.

More details about the VAN can be found in [**Visual Attention Network**](https://arxiv.org/abs/2202.09741).

## Citation

```bib
@article{guo2022visual,
  title={Visual Attention Network},
  author={Guo, Meng-Hao and Lu, Cheng-Ze and Liu, Zheng-Ning and Cheng, Ming-Ming and Hu, Shi-Min},
  journal={arXiv preprint arXiv:2202.09741},
  year={2022}
}
```
## Preparation

Install MMDetection and download COCO according to the guidelines in MMDetection.

## Requirements

We recommend following the official instructions for installing the Open-MMLab libraries, using mim. Otherwise, version mismatches are likely.

```
pip install wandb timm pycocotools openmim
```

```
mim install mmcv-full==1.7.1 mmdet==2.27.0
```

As getting the correct sets of versions correct can be tricky, we provide the exact enviroments used in our tests in the `conda-freeze.txt` respective `pip-freeze.txt`.
We used [our own fork of MMDetection](https://github.com/CptCaptain/mmdetection) for our adapted copy-paste mechanism and our evaluation. (@commit 9e62e9b4f05aedf5b0b28e5c7619ef8e89097cc1)

## Training

We use 3 GPUs for training by default. Run:

```bash
./dist_train.sh /path/to/config 3
```

## Evaluation

To evaluate the model, run:

```bash
./dist_test.sh /path/to/config /path/to/checkpoint_file 3 --eval bbox
```

## FLOPs

Install torchprofile using

```bash
pip install torchprofile
```

To calculate FLOPs for a model, run:

```bash
bash tools/flops.sh /path/to/config --shape 1333 800
```

In our evaluation, we used the updated analysis tools of our MMDetection fork.

## Acknowledgment

Our implementation is mainly based on [VAN-Segmentation](https://github.com/Visual-Attention-Network/VAN-Segmentation) and [MMDetection](https://github.com/open-mmlab/mmdetection). Thanks to the authors.

## LICENSE

This repo is under the Apache-2.0 license. For commercial use, please contact the authors.
