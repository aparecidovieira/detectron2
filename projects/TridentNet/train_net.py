#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import urllib.request
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.structures import BoxMode
import json, torch
import random
import os
from detectron2.data import DatasetCatalog, MetadataCatalog

from tridentnet import add_tridentnet_config
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_dir=output_folder)

# class CocoTrainer(DefaultTrainer):

#   @classmethod
#   def build_evaluator(cls, cfg, dataset_name, output_folder=None):

#     if output_folder is None:
#         os.makedirs("coco_eval", exist_ok=True)
#         output_folder = "coco_eval"

#     return COCOEvaluator(dataset_name, cfg, False, output_folder)


class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)


weights_catalog = {'R-50-GN': "https://dl.fbaipublicfiles.com/detectron/ImageNetPretrained/47261647/",
                'R-50.pkl': "https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/"}


def custom_mapper(dataset_dict):
    # it will be modified by code below
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    transform_list = [
        T.Resize((512, 512)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomRotation(angle=[90, 90]),
        T.RandomLighting(0.7),
        # T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
    ]
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(
        image.transpose(2, 0, 1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        # if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


def setup(args):
    """
    Create configs and perform basic setups.
    """
    # cfg = get_cfg()
    dataset_path = '/raid/cesar_workspace/cesar_workspace/Object_Detection/Detectron2/detectron2/detectron2/data/Datasets/'
    train = dataset_path + "up_trees_train_2021"
    val = dataset_path + "up_trees_val_2021"
    train_dataset = train  # cfg.DATASETS.TRAIN
    val_dataset = val  # cfg.DATASETS.TEST
    # print(cfg.DATASETS.TRAIN, 'eee')
    dic_marks_path = dataset_path + "up_trees_labels.json"
    datasets_dic = {'train': train_dataset, 'val': val_dataset}
    dic_marks = {'0':'up_tree'}
#     with open(dic_marks_path, 'w') as out:
#         json.dump(dic_marks, out)
    with open(dic_marks_path, 'r') as out:
        dic_marks = json.load(out)
    # cat_ids = {label:key for key, label in dic_marks.items()}
    classes = [label for key, label in dic_marks.items()]

    def get_board_dicts(imgdir):
        json_file = imgdir + '.json'  # Fetch the json file
        print(json_file)
        with open(json_file) as f:
            dataset_dicts = json.load(f)
        for i in dataset_dicts:
            filename = i["file_name"]
            for j in i["annotations"]:
                # Setting the required Box Mode
                j["bbox_mode"] = BoxMode.XYWH_ABS
                j["category_id"] = int(j["category_id"])
        return dataset_dicts

    # Registering the Dataset

    for d in ['val', 'train']:
        # print(datasets_dic[d])
        dataset_name = os.path.basename(datasets_dic[d])
        print(dataset_name)

        DatasetCatalog.register(dataset_name, lambda d=d: get_board_dicts(
            datasets_dic[d]))
        MetadataCatalog.get(dataset_name).set(thing_classes=classes)

    train_name = os.path.basename(datasets_dic['train'])
    val_name = os.path.basename(datasets_dic['val'])
    print(train_name, val_name)
    board_metadata = MetadataCatalog.get(train_name)
    dataset_dicts = get_board_dicts(train_dataset)
    n_imgs = len(dataset_dicts)
    dataset_dicts = get_board_dicts(val_dataset)
    n_imgs_val = len(dataset_dicts)
    print('Number of images on training data is :', n_imgs)
    cfg = get_cfg()
    add_tridentnet_config(cfg)

#     cfg.DATALOADER.NUM_WORKERS = 2

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

    # No. of iterations after which the Validation Set is evaluated.
    cfg.TEST.EVAL_PERIOD = (n_imgs//cfg.SOLVER.IMS_PER_BATCH)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)

    path = '/raid/cesar_workspace/cesar_workspace/Object_Detection/Detectron/detectron2/ImageNetPretrained/MSRA/'
    os.makedirs(path, exist_ok=True)
    backbone = os.path.basename(cfg.MODEL.WEIGHTS)
    print('Number of images on training data is :', n_imgs, n_imgs_val)
    backbone += '.pkl' if '.pkl' not in backbone else ''
    weight = path + backbone
    print(weight)
    if not os.path.isfile(weight):
        print("Downloading ImageNet weights")
        url_weights = weights_catalog[backbone] + backbone 
        urllib.request.urlretrieve(url_weights, weight)

    cfg.MODEL.WEIGHTS = weight
    print(weight)

    # cfg.OUTPUT_DIR = './output_%s_X-101_b/'%accr
    print(cfg, '~~ I dedicate this to Shadow Moon ~~')

    default_setup(cfg, args)
    cfg.freeze()
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
