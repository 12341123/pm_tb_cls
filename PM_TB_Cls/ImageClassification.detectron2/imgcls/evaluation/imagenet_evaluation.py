import itertools
import json
import logging
from collections import OrderedDict
import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.evaluation.evaluator import DatasetEvaluator
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
logger = logging.getLogger("detectron2.classification_eval")
import json
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

class ImageNetEvaluator(DatasetEvaluator):

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger("detectron2.evaluation.PMTB_evaluation")

        self._metadata = MetadataCatalog.get(dataset_name)

        # json_file = PathManager.get_local_path(self._metadata.json_file)
        # self._gt = json.load(open(json_file))

    def reset(self):
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # prediction = {"image_id": input["image_id"]}
            prediction ={}
            prediction["gt"]=input['label'][:2]
            prediction["pred"] = output["pred_classes"].to(self._cpu_device)
            prediction['loss'] = output['loss']
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning(
                "[ImageNetEvaluator] Did not receive valid predictions.")
            return {}

        target = []
        pred = []
        loss = []
        for p in predictions:
            if p['gt'][:2] == [1,0]:
                target.append([1,0,0])
            elif p['gt'][:2] == [0,1]:
                target.append([0, 1, 0])
            else:
                target.append([0, 0, 1])
            # target.append(p['gt'])
            pred.append(p['pred'])
            loss.append(p['loss'])

        # res = []
        # for i in range(len(target)):
        #     record = {
        #         'gt':np.asarray(target[i]).tolist(),
        #         'pred':np.asarray(pred[i]).tolist(),
        #     }
        #     res.append(record)
        # with open('/ssd2/wangzd/pred.json','w') as f:
        #     json.dump(res,f)



        all_gts =np.asarray(target)
        all_preds = np.array([np.array(i) for i in pred])
        all_loss = np.asarray(loss)
        myMetic = Metric(all_preds, all_gts)
        average_auc, auc_list = myMetic.auROC()
        # print(average_auc)
        # print(np.mean(all_loss))
        logger.info(average_auc)
        logger.info(np.mean(all_loss))
        # classes_names = ['PM',"TB"]
        # score_thresholds = [0.45,0.5]
        # for i in range(all_gts.shape[1]):
        #     cls_report = classification_report(y_true=all_gts[:, i], y_pred=np.where(all_preds[:, i] >score_thresholds[i], 1, 0),
        #                           target_names=['Negative', classes_names[i]])
        #     logger.info(f'\{classes_names[i]}classification report:\n{cls_report}')
        #     c_mat = confusion_matrix(y_true=all_gts[:, i],y_pred=np.where(all_preds[:, i] > score_thresholds[i], 1, 0),labels=[0,1])
        #     logger.info(f'\{classes_names[i]}confusion matrix:\n{c_mat}')
        # result ={
        #     'PM_AUC':auc_list[0],
        #     'TB_AUC':auc_list[1],
        #     'Average_AUC':average_auc,
        # }
        # return result
        return

