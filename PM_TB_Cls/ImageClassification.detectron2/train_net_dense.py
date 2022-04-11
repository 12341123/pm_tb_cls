import time
from vision.config import get_cfg
from vision.build import build_dataloader  # , build_model
from vision.utils import (
    inference_context,
    CustomizedCommonMetricPrinter,
    smooth_mask,
)
import json
import cv2
import datetime
from torch.nn.parallel import DistributedDataParallel
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.engine import default_argument_parser, launch, default_setup
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.data import MetadataCatalog
from detectron2.modeling.meta_arch import build_model
import numpy as np
from vision.loss import SoftDiceLoss
from detectron2.utils.events import (
    # CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)
import logging
import os
from collections import OrderedDict
import torch
from sklearn.metrics import classification_report, confusion_matrix
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader, DatasetCatalog, build_batch_data_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
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

from imgcls.config import get_cfg
import imgcls.modeling
from imgcls.data import DatasetMapper,classification_utils
from imgcls.evaluation.imagenet_evaluation import ImageNetEvaluator
import sklearn
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import eigs
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
import itertools
from detectron2.solver.build import maybe_add_gradient_clipping, get_default_optimizer_params
from detectron2.config import CfgNode as CN

logger = logging.getLogger("detectron2.vision")
# np.set_printoptions(threshold='nan')
class Metric(object):
    def __init__(self, output, label):
        self.output = output  # prediction label matric
        self.label = label  # true  label matric

    def accuracy_subset(self, threash=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy

    def accuracy_mean(self, threash=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        accuracy = np.mean(np.equal(y_true, y_pred))
        return accuracy

    def accuracy_multiclass(self):
        y_pred = self.output
        y_true = self.label
        accuracy = accuracy_score(np.argmax(y_pred, 1), np.argmax(y_true, 1))
        return accuracy

    def micfscore(self, threash=0.5, type='micro'):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def macfscore(self, threash=0.5, type='macro'):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        return f1_score(y_pred, y_true, average=type)

    def hamming_distance(self, threash=0.5):
        y_pred = self.output
        y_true = self.label
        y_pred = np.where(y_pred > threash, 1, 0)
        return hamming_loss(y_true, y_pred)

    def fscore_class(self, type='micro'):
        y_pred = self.output
        y_true = self.label
        return f1_score(np.argmax(y_pred, 1), np.argmax(y_true, 1), average=type)

    def auROC(self):
        y_pred = self.output
        y_true = self.label
        row, col = y_true.shape
        temp = []
        ROC = 0
        for i in range(col):
            ROC = roc_auc_score(y_true[:, i], y_pred[:, i], average='micro', sample_weight=None)
            print("%d th AUROC: %f" % (i, ROC))
            temp.append(ROC)
        for i in range(col):
            ROC += float(temp[i])
        return ROC / (col + 1) , temp

    def MacroAUC(self):
        y_pred = self.output  # num_instance*num_label
        y_true = self.label  # num_instance*num_label
        num_instance, num_class = y_pred.shape
        count = np.zeros((num_class, 1))  # store the number of postive instance'score>negative instance'score
        num_P_instance = np.zeros((num_class, 1))  # number of positive instance for every label
        num_N_instance = np.zeros((num_class, 1))
        auc = np.zeros((num_class, 1))  # for each label
        count_valid_label = 0
        for i in range(num_class):  # 第i类
            num_P_instance[i, 0] = sum(y_true[:, i] == 1)  # label,,test_target
            num_N_instance[i, 0] = num_instance - num_P_instance[i, 0]
            # exclude the label on which all instances are positive or negative,
            # leading to num_P_instance(i,1) or num_N_instance(i,1) is zero
            if num_P_instance[i, 0] == 0 or num_N_instance[i, 0] == 0:
                auc[i, 0] = 0
                count_valid_label = count_valid_label + 1
            else:

                temp_P_Outputs = np.zeros((int(num_P_instance[i, 0]), num_class))
                temp_N_Outputs = np.zeros((int(num_N_instance[i, 0]), num_class))
                #
                temp_P_Outputs[:, i] = y_pred[y_true[:, i] == 1, i]
                temp_N_Outputs[:, i] = y_pred[y_true[:, i] == 0, i]
                for m in range(int(num_P_instance[i, 0])):
                    for n in range(int(num_N_instance[i, 0])):
                        if (temp_P_Outputs[m, i] > temp_N_Outputs[n, i]):
                            count[i, 0] = count[i, 0] + 1
                        elif (temp_P_Outputs[m, i] == temp_N_Outputs[n, i]):
                            count[i, 0] = count[i, 0] + 0.5

                auc[i, 0] = count[i, 0] / (num_P_instance[i, 0] * num_N_instance[i, 0])
        macroAUC1 = sum(auc) / (num_class - count_valid_label)
        return float(macroAUC1), auc
class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))

    @classmethod
    def build_optimizer(cls, cfg, model):
        params = get_default_optimizer_params(
            model,
            base_lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
            bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
            weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
        )

        def maybe_add_full_model_gradient_clipping(optim):  # optim: the optimizer class
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                    cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                    and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                    and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=0.9,
                nesterov=cfg.SOLVER.NESTEROV,
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        elif optimizer_type == "AdamW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR, betas=(0.9, 0.999),
                weight_decay=cfg.SOLVER.WEIGHT_DECAY,
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        return optimizer

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(
                dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(
                dataset_name, output_folder))
        # elif evaluator_type == "cityscapes":
        #     assert (
        #         torch.cuda.device_count() >= comm.get_rank()
        #     ), "CityscapesEvaluator currently do not work with multiple machines."
        #     return CityscapesEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        elif evaluator_type == "imagenet":
            return ImageNetEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(
                    cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
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
def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
def do_test_multilabel(cfg, model, iteration=-1):
    # num_labels = 6
    # num_classes = 2
    image_ids = []
    preds_list, labels_list = [], []
    data_loader = build_dataloader(cfg, is_train=False)
    # with inference_context(model), torch.no_grad():
    with inference_context(model), torch.no_grad():
        for inputs in data_loader:
            results = model(inputs)
            for i in range(len(results)):
                preds = results[i]['pred_classes'][::-1].to('cpu')  # list
                preds_list.append(np.array(preds))
                labels = results[i]['Ground_Truth_classes']
            ids = [x["file_name"] for x in inputs]
            image_ids.extend(ids)
    comm.synchronize()
    total_preds, total_labels = np.array(preds_list), np.array(labels_list)
    all_preds = comm.gather(total_preds,dst=0)
    all_gts = comm.gather(total_labels,dst=0)
    all_ids =comm.gather(image_ids,dst=0)
    if comm.is_main_process():
        all_preds = np.array(list(itertools.chain(*all_preds)))
        all_gts = np.array(list(itertools.chain(*all_gts)))
        all_ids = list(itertools.chain(*all_ids))
        myMetic = Metric(all_preds, all_gts)
        average_auc,auc_list = myMetic.auROC()
        logger.info(f'average_auc:\n{average_auc}')
        logger.info(f'auc_per_class:\n{auc_list}')
        classes_names = ['Cardiomegaly','Edema','Consolidation','Pneumonia','Atelectasis','Pleural Effusion']
        for i in range(all_gts.shape[1]):
            cls_report = classification_report(y_true=all_gts[:, i], y_pred=np.where(all_preds[:, i] > cfg.MODEL.BCE.SCORETHRESHOLD, 1, 0),
                                  target_names=['Negative', classes_names[i]])
            logger.info(f'\{classes_names[i]}classification report:\n{cls_report}')
            c_mat = confusion_matrix(y_true=all_gts[:, i],y_pred=np.where(all_preds[:, i] > cfg.MODEL.BCE.SCORETHRESHOLD, 1, 0),labels=[0,1])
            logger.info(f'\{classes_names[i]}confusion matrix:\n{c_mat}')
        inference_dir = os.path.join(cfg.OUTPUT_DIR, 'inference')
        if not os.path.exists(inference_dir):
            os.makedirs(inference_dir)
        result = []
        for id , gt , pred in zip(all_ids,all_gts,all_preds):
            result.append({
                "image_file":id,
                "GT":gt.tolist(),
                "Prediction":pred.tolist()
            })
        results_file = os.path.join(inference_dir, f'results_{iteration}.json')
        logger.info(f'Saving results to {results_file}')
        with open(results_file, 'w') as f:
            json.dump(result, f)
def do_test_pm_tb_cls(cfg, model, iteration=-1):
    image_ids = []
    preds_list, labels_list = [], []
    data_loader = build_dataloader(cfg, is_train=False)
    # with inference_context(model), torch.no_grad():
    with inference_context(model), torch.no_grad():
        for inputs in data_loader:
            results = model(inputs)
            for i in range(len(results)):
                preds = results[i]['pred_classes'].to('cpu')  # list
                preds_list.append(np.array(preds))
                labels2 = results[i]['Ground_Truth_classes']
                labels = []
                for i in labels2[:2]:
                    if i<0.5:
                        labels.append(0)
                    else:
                        labels.append(1)
                labels_list.append(np.array(labels[::-1]))
            ids = [x["file_name"] for x in inputs]
            image_ids.extend(ids)
    comm.synchronize()
    total_preds, total_labels = np.array(preds_list), np.array(labels_list)
    all_preds = comm.gather(total_preds,dst=0)
    all_gts = comm.gather(total_labels,dst=0)
    all_ids =comm.gather(image_ids,dst=0)
    if comm.is_main_process():
        all_preds = np.array(list(itertools.chain(*all_preds)))
        all_gts = np.array(list(itertools.chain(*all_gts)))
        all_ids = list(itertools.chain(*all_ids))
        myMetic = Metric(all_preds, all_gts)
        average_auc,auc_list = myMetic.auROC()
        logger.info(f'average_auc:\n{average_auc}')
        logger.info(f'auc_per_class:\n{auc_list}')
        classes_names = ['TB',"PM"]
        for i in range(all_gts.shape[1]):
            cls_report = classification_report(y_true=all_gts[:, i], y_pred=np.where(all_preds[:, i] >= cfg.MODEL.BCE.SCORETHRESHOLD, 1, 0),
                                  target_names=['Negative', classes_names[i]])
            logger.info(f'\{classes_names[i]}classification report:\n{cls_report}')
            c_mat = confusion_matrix(y_true=all_gts[:, i],y_pred=np.where(all_preds[:, i] > cfg.MODEL.BCE.SCORETHRESHOLD, 1, 0),labels=[0,1])
            logger.info(f'\{classes_names[i]}confusion matrix:\n{c_mat}')
        inference_dir = os.path.join(cfg.OUTPUT_DIR, 'inference')
        if not os.path.exists(inference_dir):
            os.makedirs(inference_dir)
        result = []
        for id , gt , pred in zip(all_ids,all_gts,all_preds):
            result.append({
                "image_file":id,
                "GT":gt.tolist(),
                "Prediction":pred.tolist()
            })
        results_file = os.path.join(inference_dir, f'results_{iteration}.json')
        logger.info(f'Saving results to {results_file}')
        with open(results_file, 'w') as f:
            json.dump(result, f)

def main(args):
    with open('/ssd2/wangzd/194/1490_train_3.json') as train:
        train_dict194 = json.load(train)
    with open('/ssd2/wangzd/206/206_bbox_train.json') as train:
        train_dict206 = json.load(train)
    with open('/ssd2/wangzd/206/aug_image.json') as train:
        train_dict206_aug = json.load(train)
    with open('/ssd2/wangzd/194/194_bbox_val.json') as val:
        val_dict194 = json.load(val)
    with open('/ssd2/wangzd/206/206_bbox_val.json') as val:
        val_dict206 = json.load(val)
    with open("/ssd2/wangzd/FP_train.json") as fp:
        fp_dict_train = json.load(fp)
    with open("/ssd2/wangzd/FP_val.json") as fp:
        fp_dict_val= json.load(fp)
    with open("/ssd2/wangzd/tuixiang.json") as fp:
        tuixiang= json.load(fp)
    with open('/ssd2/wangzd/vbd_fp_train.json') as fp:
        vbd_fp_train = json.load(fp)
    with open('/ssd2/wangzd/vbd_fp_val.json') as fp:
        vbd_fp_val = json.load(fp)
    with open('/ssd2/wangzd/cangzhou_FP_train.json') as c:
        cangzhou_train = json.load(c)
    with open('/ssd2/wangzd/cangzhou_FP_val.json') as c:
        cangzhou_val = json.load(c)
    with open('/ssd2/wangzd/train_fp.json') as c:
        train_fp = json.load(c)

    all_val=val_dict194+val_dict206+vbd_fp_val+cangzhou_val
    def get_train_fp():
        return train_fp
    def get_train_cangzhou():
        return cangzhou_train
    def get_194_206():
        return val_dict206+val_dict194
    def get_all_val():
        return all_val
    def get_train194():
        return train_dict194
    def get_val194():
        return val_dict194
    def get_train206():
        return train_dict206
    def get_train206_aug():
        return train_dict206_aug
    def get_val206():
        return val_dict206
    def get_fp_train():
        return vbd_fp_train
    def get_fp_val():
        return vbd_fp_val
    def get_tuixiang():
        return tuixiang
    def get_obj_fp_train():
        return fp_dict_train
    def get_obj_fp_val():
        return fp_dict_val

    DatasetCatalog.register('obj_train', get_obj_fp_train)
    DatasetCatalog.register('194_train',get_train194)
    DatasetCatalog.register('206_train',get_train206)
    DatasetCatalog.register('206_train_aug',get_train206_aug)
    DatasetCatalog.register('194_val',get_val194)
    DatasetCatalog.register('206_val',get_val206)
    DatasetCatalog.register('all_val', get_all_val)
    DatasetCatalog.register('194_206', get_194_206)
    DatasetCatalog.register('fp_vbd',get_fp_train)
    DatasetCatalog.register('fp_val',get_fp_val)
    DatasetCatalog.register('fp_cangzhou', get_train_cangzhou)
    DatasetCatalog.register('tuixiang',get_tuixiang)
    DatasetCatalog.register('train_fp', get_train_fp)

    MetadataCatalog.get('tuixiang').evaluator_type='imagenet'
    MetadataCatalog.get('194_val').evaluator_type='imagenet'
    MetadataCatalog.get('206_val').evaluator_type='imagenet'
    MetadataCatalog.get('fp_val').evaluator_type='imagenet'
    MetadataCatalog.get('all_val').evaluator_type = 'imagenet'
    MetadataCatalog.get('194_206').evaluator_type = 'imagenet'

    cfg = setup(args)
    if args.eval_only:
        # cfg.defrost()
        # cfg.MODEL.WEIGHTS = '/ssd2/wangzd/pm_tb_3/model_0060999.pth'
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()
if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args: ", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
