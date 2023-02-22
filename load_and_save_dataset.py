# Copyright (c) OpenMMLab. All rights reserved.
# adapted by me to save a validation set of 
# the results from a MultiImageMixDataset

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
import json

import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet.core.utils import mask2ndarray
from mmdet.core.visualization import imshow_det_bboxes
from mmdet.datasets.builder import build_dataset
from mmdet.utils import replace_cfg_vals, update_data_root

import cv2

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

categories = {obj_class: i for i, obj_class in enumerate(CLASSES)}



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        required=True,
        help='Path to store the generated images and annotations (in subfolders)')
    parser.add_argument('--not-show', default=False, action='store_true')
    args = parser.parse_args()
    return args


def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }
    return coco_format


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def retrieve_data_cfg(config_path, skip_type):

    def skip_pipeline_steps(config):
        config['pipeline'] = [ x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    # update data root according to MMDET_DATASETS
    update_data_root(cfg)

    train_data_cfg = cfg.data.train
    while 'dataset' in train_data_cfg and train_data_cfg[
        'type'] != 'MultiImageMixDataset':
        train_data_cfg = train_data_cfg['dataset']

    if isinstance(train_data_cfg, Sequence):
        [skip_pipeline_steps(c) for c in train_data_cfg]
    else:
        skip_pipeline_steps(train_data_cfg)

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    if 'gt_semantic_seg' in cfg.train_pipeline[-1]['keys']:
        cfg.data.train.pipeline = [
            p for p in cfg.data.train.pipeline if p['type'] != 'SegRescale'
        ]
    # TODO create new cfg with MultiImageMixDataset and val split for this task!
    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))

    coco = get_coco_json_format()
    images, annotations = [], []
    annotation_id = 0
    image_id = 0

    for i, item in enumerate(dataset):
        if i > 5:
            break
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None

        gt_bboxes = item['gt_bboxes']
        gt_labels = item['gt_labels']
        gt_masks = item.get('gt_masks', None)
        if gt_masks is not None:
            gt_masks = mask2ndarray(gt_masks)

        # Step One: save image 
        out_file = f'{args.output_dir}/img/{image_id:05}.jpg'
        cv2.imwrite(out_file, item['img'])

        # Step Two: generate image info
        img_info = {
            'filename': out_file,
            'width': item['img'].shape[0],
            'height': item['img'].shape[1],
            'id': i,
        }
        images.append(img_info)

        zipper = zip(
            item['gt_bboxes'],
            item['gt_labels'],
            item['gt_masks'],
        )

        # Step three: generate annotation infos
        for i, (bbox, label, mask) in enumerate(zipper):
            ann_info = {
                'image_id': image_id,
                'category_id': label,
                'bbox': bbox,
                'segmentation': mask,
                'id': annotation_id,
            }
            annotations.append(ann_info)
            annotation_id += 1

        progress_bar.update()
        image_id += 1

    coco['images'] = images
    coco['annotations'] = annotations
    coco['categories'] = create_category_annotation(categories)
    
    ann_path = Path(f'{args.output_dir}/val.json')
    with open(ann_path, 'w') as ann_file:
        json.dump(coco, ann_file, cls=NpEncoder)


if __name__ == '__main__':
    main()

