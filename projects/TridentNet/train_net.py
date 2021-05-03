#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
TridentNet Training Script.

This script is a simplified version of the training script in detectron2/tools.
"""

import os
import urllib.request

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
import torch
from tridentnet import add_tridentnet_config


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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    from detectron2.structures import BoxMode
    import json, random, os
#     '0' 152 
    dic_marks = {'0': 'No_right_turn',
    '1':'No_left_turn',
    '2':'Prohibited_straight',
    '3':'Straight_left_turn_prohibited',
    '4':'Go_straight_not_turn_right',
    '5':'Left_right_turn_prohibited',
    '6':'No_U-turn',
    '7':'Speed_limit',
    '8':'Speed_limit_child_zone',
    '9':'Concession',
    '10':'Crosswalk',
    '11':'Bike_Priority_Road',
    '12':'Turn_left',
    '13':'Go_straight',
    '14':'Turn_right',
    '15':'Turn_left_go_straight',
    '16':'Go_straight_turn_right',
    '17':'U-turn',
    '18':'U-turn-left_turn',
    '19':'Unprotected_Left_Turn'}
#     dic_marks = {'0': 'Crosswalk'}
    # cat_ids = {label:key for key, label in dic_marks.items()}
    classes = [label for key, label in dic_marks.items()]
    base = '/media/hdd/cesar_workspace/Object_Detection/Detectron2/detectron2/detectron2/data/Datasets/'

    def get_board_dicts(imgdir):
        json_file = imgdir + '.json'#+"/dataset.json" #Fetch the json file
        print(json_file)
        with open(json_file) as f:
            dataset_dicts = json.load(f)
        for i in dataset_dicts:
            filename = i["file_name"] 
            for j in i["annotations"]:
                j["bbox_mode"] = BoxMode.XYWH_ABS #Setting the required Box Mode
                j["category_id"] = int(j["category_id"])
        return dataset_dicts


    from detectron2.data import DatasetCatalog, MetadataCatalog
    
    #Registering the Dataset
    data_type = '20k_NGI'
    for d in ['val', 'train']:
        DatasetCatalog.register("lanes_%s_%s"%(d, data_type), lambda d=d: get_board_dicts(base + "lanes_%s_%s"%(d, data_type)))
        MetadataCatalog.get("lanes_%s_%s"%(d, data_type)).set(thing_classes=classes)
        
    board_metadata = MetadataCatalog.get("lanes_train_%s"%data_type)
    dataset_dicts = get_board_dicts(base + "lanes_train_%s"%data_type)
    n_imgs = len(dataset_dicts)
    
    dataset_dicts = get_board_dicts(base + "lanes_val_%s"%data_type)
    n_img_test = len(dataset_dicts)
    
    print('Number of images on training data is :',n_imgs, n_img_test)
    cfg = get_cfg()
    add_tridentnet_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.DATALOADER.NUM_WORKERS = 2
#     cfg.SOLVER.IMS_PER_BATCH = 2
#     cfg.SOLVER.MAX_ITER = 00
#     cfg.SOLVER.BASE_LR = 0.002
    
#     cfg.SOLVER.MAX_ITER = int(n_imgs//cfg.SOLVER.IMS_PER_BATCH * 80)  #No. of iterations   
    cfg.DATASETS.TRAIN = ("lanes_train_%s"%(data_type),)
    cfg.DATASETS.TEST = ("lanes_val_%s"%(data_type),)
    cfg.TEST.EVAL_PERIOD =  int(n_imgs//cfg.SOLVER.IMS_PER_BATCH) # No. of iterations after which the Validation Set is evaluated. 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  
#     cfg.NUM_GPUS = 2
#     cfg.MODEL.DEVICE='cuda:1'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes) # No. of classes = [HINDI, ENGLISH, OTHER]
    print(cfg.SOLVER.MAX_ITER)
#     return None
    path = '/media/hdd/cesar_workspace/Object_Detection/Detectron2/detectron2/ImageNetPretrained/MSRA/'
#     if 'X-101' in path or '152' in path:
#         path+='FAIR/'
    os.makedirs(path, exist_ok=True)
    backbone = os.path.basename(cfg.MODEL.WEIGHTS)
    print('Number of images on training data is :',n_imgs, n_img_test)
    
#    "detectron2://ImageNetPretrained/MSRA/R-101.pkl"
    weight = path + backbone
#     print(weight)
#     if not os.path.isfile(weight):
#         url_weights = 'https://dl.fbaipublicfiles.com/detectron2/' + cfg.MODEL.WEIGHTS.replace('detectron2://', '')
#         print(url_weights)      
#         urllib.request.urlretrieve(url_weights, weight)
# #         url_weights = 'https://dl.fbaipublicfiles.com/detectron2/ImageNetPretrained/MSRA/R-50.pkl'
# #     weight = '/media/hdd/cesar_workspace/Object_Detection/Detectron2/detectron2/projects/TridentNet/output-101/model_0179999.pth'
#     weight = '/media/hdd/cesar_workspace/Object_Detection/Detectron2/detectron2/ImageNetPretrained/MSRA/R-50-GN.pkl'
#     cfg.MODEL.WEIGHTS = weight
    print(cfg.MODEL.WEIGHTS, 'Loading weights like a Jedi')
#     cfg.MODEL.WEIGHTS = './output_20k_X/model_final.pth'

#     cfg.MODEL.WEIGHTS = '/media/hdd/cesar_workspace/Object_Detection/Detectron2/detectron2/ImageNetPretrained/MSRA/R-50.pkl'
#     cfg.SOLVER.STEPS: (3000, 6000)
#     cfg.SOLVER.MAX_ITER = 150000
        
#     cfg.OUTPUT_DIR = './output-101/'
#     cfg.OUTPUT_DIR = './output_20k_X-152/'#%data_type
#     cfg.MODEL.WEIGHTS = cfg.OUTPUT_DIR + 'model_0144999.pth'
#     args.resume = True
    default_setup(cfg, args)    
    cfg.freeze()
   

    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
#         PATH = "./entire_model.pt"
        
# # Save
#         torch.save(model, PATH)
        
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)

        return res
    
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
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
