# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import glob
import multiprocessing as mp
import numpy as np
import os
import tempfile
import time
import warnings
import cv2
import tqdm
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
# from predictor import VisualizationDemo
# from swint import add_swint_config
# from swint import add_swint_config
import matplotlib.pyplot as plt
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import CfgNode as CN
from imgcls.data import DatasetMapper,classification_utils
from imgcls.config import get_cfg
from detectron2.modeling import build_model
import imgcls.modeling
from imgcls.data import DatasetMapper

import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import detectron2.data.transforms as T
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
import atexit
import logging
import os
import json
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import cv2
import pydicom
import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger
from convert_dicom_v2 import convert_dicom_v2
# constants
WINDOW_NAME = "COCO detections"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def add_cfg(cfg):
    cfg.MODEL.BCE=CN()
    cfg.MODEL.BCE.INPUTFEATURESIZE = 1024
    cfg.MODEL.BCE.MLPFEATURESIZE =512
    cfg.MODEL.BCE.BCECLASS = 2
    cfg.INPUT.KEEP_ASPECT_RATIO = True
    cfg.MODEL.BCE.SCORETHRESHOLD = 0.5
    # cfg.MODEL.RESNETS.DEPTH = 101
    cfg.SOLVER.OPTIMIZER = "AdamW"
    cfg.MODEL.DENSENET = CN()
    cfg.MODEL.DENSENET.DEPTH = 121
    cfg.MODEL.DENSENET.OUT_FEATURES = ['dense4']
    cfg.MODEL.CLASSIFIER = CN()
    cfg.MODEL.CLASSIFIER.PRETRAINED = True
    cfg.MODEL.CLSNET=CN()
    cfg.MODEL.CLSNET.ENABLE= True
    cfg.MODEL.CLSNET.NUM_CLASSES= 2
    cfg.MODEL.CLSNET.INPUT_SIZE= 224
    cfg.MODEL.CLSNET.IN_FEATURES =["dense4"]
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg
def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser
def test_opencv_video_format(codec, file_ext):
    with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
        filename = os.path.join(dir, "test_file" + file_ext)
        writer = cv2.VideoWriter(
            filename=filename,
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(30),
            frameSize=(10, 10),
            isColor=True,
        )
        [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
        writer.release()
        if os.path.isfile(filename):
            return True
        return False
class Predictor:
    def __init__(self, cfg, resume=True, device=None):
        self.cfg = cfg.clone()
        if resume:
            with open(os.path.join(self.cfg.OUTPUT_DIR, "last_checkpoint"), "r") as f:
                chkp_name = f.read().strip()
                chkp_path = os.path.join(cfg.OUTPUT_DIR, chkp_name)
            self.cfg.merge_from_list([
                "MODEL.WEIGHTS", chkp_path,
                "MODEL.DEVICE", cfg.MODEL.DEVICE if device is None else device,
            ])

        self.device = device
        self.model = build_model(self.cfg)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        print(f"Loading weights: {self.cfg.MODEL.WEIGHTS} ...")
        ckpt = checkpointer.load(self.cfg.MODEL.WEIGHTS)
        self.iteration = ckpt.get("iteration", -1)
        self.aug = T.AugmentationList(classification_utils.build_transform_gen(cfg, is_train=False))

    def __call__(self, original_image):
        assert original_image.ndim == 3, f"original image should be in HWC format"
        with torch.no_grad():
            height, width = original_image.shape[:2]
            aug_input = T.AugInput(image=original_image)
            tfms = self.aug(aug_input)
            image = torch.as_tensor(
                np.ascontiguousarray(aug_input.image.transpose((2, 0, 1)).astype("float32"))
            )

            inputs = {"image": image, "height": height, "width": width,"label":[],"file_name":'_'}
            predictions = self.model([inputs])[0]

            return predictions
def get_prediction(predictor,image):
    predictions = predictor(image)
    pred = predictions['pred_classes'].cpu().numpy()
    return pred

mp.set_start_method("spawn", force=True)
args = get_parser().parse_args()
setup_logger(name="fvcore")
logger = setup_logger()
logger.info("Arguments: " + str(args))
cfg = get_cfg()
# cfg = setup_cfg(args)
add_cfg(cfg)
cfg.merge_from_file("/ssd2/wangzd/pm_tb_4_9_2_finetune/config.yaml")
cfg.MODEL.WEIGHTS = "/ssd2/wangzd/pm_tb_4_9_2_finetune/model_0000999.pth"
predictor = Predictor(cfg,resume=False)
#Need to give an image
json_path = '/data142T/users/kesicheng/eval_data/result/172_result.json'
with open(json_path) as f:
    imagelist = json.load(f)
for file in imagelist:
    dicompath = file['file_name']
    ds = pydicom.dcmread(dicompath)
    image = convert_dicom_v2(ds,full_range=True)
    bbox = file["lung_bbox"]
    abnor = file['bbox']
    bbox_img = np.zeros_like(image)
    bbox_img[bbox[1]:bbox[3],bbox[0]:bbox[2]]=image[bbox[1]:bbox[3],bbox[0]:bbox[2]]
    if len(abnor):
        abnor_img = np.zeros_like(image)
        abnor_img[abnor[1]:abnor[3],abnor[0]:abnor[2]] = image[abnor[1]:abnor[3],abnor[0]:abnor[2]]
        aug_image = np.asarray([abnor_img,bbox_img,image])
        aug_image = aug_image.transpose([1,2,0])
        pred = get_prediction(predictor,image=aug_image)
        ans = []
        if pred[2]>=0.33:
            ans = [0,0]
        else:
            if pred[0]>=0.33:
                ans.append(1)
            else:
                ans.append(0)
            if pred[1]>=0.33:
                ans.append(1)
            else:
                ans.append(0)
        print(pred)
        file['one_hot']=ans
        file['pred'] = pred.tolist()
        file['threshold'] = [0.33,0.33,0.5]
    else:
        file['one_hot'] = [0,0]
        file['pred'] = [0,0]
        file['threshold'] = [0.33,0.33,0.5]

with open('/home1/wangzd/test/172_result.json','w') as f:
    json.dump(imagelist,f)



    # path='/data142T/users/kesicheng/tuixiang_pm_tb' #推想数据
    # path='/ssd2/wangzd/206/concate_png_bbox_2' #206
    # imagepath=sorted(os.listdir(path))
    # args.input=imagepath
    # if args.input:
    #     for path_ in tqdm.tqdm(args.input, disable=not args.output):
    #         img=cv2.imread(os.path.join(path,path_))
    #         predictions = predictor(img)
    #         pred = predictions['pred_classes'].cpu().numpy()
    #         ans =[]
    #         if pred[0]>=0.5:
    #             ans.append(1)
    #         else:
    #             ans.append(0)
    #         if pred[1]>=0.55:
    #             ans.append(1)
    #         else:
    #             ans.append(0)
    #         ans.append(pred)
    #         predictions['pred_classes']=ans #第一个位置是肺炎置信度，第二个位置是肺结核置信度
    #         print(predictions)





