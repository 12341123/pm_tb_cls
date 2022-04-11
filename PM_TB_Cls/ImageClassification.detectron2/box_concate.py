import os
import cv2
import json
import torch
import torchvision
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from detectron2.structures import ImageList
from detectron2.data import transforms as T
from detectron2.data.transforms import Augmentation
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.transforms import ResizeShortestEdge
from fvcore.transforms.transform import Transform, NoOpTransform


class RescaleTransform(Transform):
    def __init__(self, scale, rescale_mask=False):
        super(RescaleTransform, self).__init__()
        self.scale = scale
        self.rescale_mask = rescale_mask

    def apply_image(self, img: np.ndarray) -> np.ndarray:
        return img * float(self.scale)

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        if self.rescale_mask:
            return self.apply_image(segmentation)
        else:
            return segmentation

    def inverse(self) -> Transform:
        return NoOpTransform()
class Rescale(Augmentation):
    def __init__(self, scale: float, rescale_mask: bool = False):
        self.scale = scale
        self.rescale_mask = rescale_mask

    def get_transform(self, image) -> Transform:
        return RescaleTransform(self.scale, self.rescale_mask)
def build_augmentation():
    augmentation = []
    min_size, max_size = 800, 1333
    sample_style = 'choice'
    resize = ResizeShortestEdge(min_size, max_size, sample_style)
    augmentation.append(resize)

    # rescale the pixel range to [0, 1]

    rescale = Rescale(scale=1 / 255.0)
    augmentation.append(rescale)

    return augmentation
class SemanticSegmentation(torch.nn.Module):
    def __init__(self):
        super(SemanticSegmentation, self).__init__()
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1))
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, num_classes=1,
                                                                         pretrained_backbone=False)
    @property
    def device(self):
        return self.pixel_mean.device
    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)
    def inference(self, batched_inputs) -> list:
        assert not self.training
        images = self.preprocess_image(batched_inputs)
        logits = self.model(images.tensor)["out"]
        results = logits.sigmoid()
        processed_results = []
        for result, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
            original_height = input_per_image.get("height")
            original_width = input_per_image.get("width")
            r = result[:, : image_size[0], : image_size[1]].expand(1, -1, -1, -1)
            r = F.interpolate(
                r, size=(original_height, original_width), mode="bilinear",
                align_corners=False
            )[0]
            processed_results.append({"sem_seg": r})

        return processed_results
    def preprocess_image(self, batched_inputs) -> ImageList:
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images)

        return images
class Predictor:
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = SemanticSegmentation()
        self.model.to(self.device)
        self.model.eval()
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(model_path)
        self.aug = T.AugmentationList(build_augmentation())
    def __call__(self, original_image):
        assert original_image.ndim == 3, f"original image should be in HWC format"
        with torch.no_grad():
            height, width = original_image.shape[:2]
            aug_input = T.AugInput(image=original_image)
            tfms = self.aug(aug_input)
            image = torch.as_tensor(
                np.ascontiguousarray(aug_input.image.transpose((2, 0, 1)).astype("float32"))
            )

            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])

            return predictions[0]
coco_dir = '/home1/wangzd/194/194_pm_tb_train.json'
data = json.load(open(coco_dir))
dataset_dict = list()
for i in range(len(data['images'])):
    img_id = data['images'][i]['id']
    file_name = data['images'][i]['file_name']
    file_name = file_name.replace('/home/sicheng.ke/data','/home1/wangzd')
    img_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    p = Predictor(model_path="/home1/wangzd/Weights/lung_20201124.pt", device="cuda")
    lung_img = p(img_rgb)["sem_seg"].cpu().numpy().squeeze() > 0.5
    lung_img = lung_img.astype(np.uint8)
    x,y,w,h = cv2.boundingRect(lung_img)
    bbox_mask = np.zeros_like(img_gray)
    bbox_mask[y:y+h,x:x+w]=1
    lung_img = img_gray * bbox_mask
    for j in range(len(data['annotations'])):
        if data['annotations'][j]['image_id'] == img_id:
            bbox = data['annotations'][j]['bbox']
            print(bbox)
            bbox_img = np.zeros_like(img_gray)
            margin = int(max(bbox[3],bbox[2])/3)
            # print(margin) bbox[0]) : int(bbox[0])+int(bbox[2])
            # print(bbox[1]-margin,bbox[1] + bbox[3]+margin,bbox[0]-margin,bbox[0],bbox[0] + bbox[2]+margin)
#             bbox_img[int(bbox[1]-margin) : int(bbox[1] + bbox[3]+margin), int(bbox[0]-margin):int(bbox[0] + bbox[2]+margin)] = \
#                 img_gray[int(bbox[1]-margin) : int(bbox[1] + bbox[3]+margin), int(bbox[0]-margin):int(bbox[0] + bbox[2]+margin)]
            bbox_img[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])] = \
                img_gray[int(bbox[1]):int(bbox[1])+int(bbox[3]), int(bbox[0]):int(bbox[0])+int(bbox[2])]
            img = np.asarray([bbox_img, lung_img, img_gray])
            img = img.transpose([1, 2, 0])
            # mask = np.asarray(data['annotations'][j]['segmentation'], dtype=np.int)
            # mask = mask.reshape(mask.shape[1] // 2, 2)
            # mask_img = np.zeros_like(img_rgb)
            # mask_img = cv2.fillPoly(mask_img, np.asarray([mask]), color=[0, 255, 0])
            # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2GRAY)
            # _, mask_img = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
            # img = np.asarray([mask_img, lung_img, img_gray])
            # img = img.transpose([1, 2, 0])
            # plt.subplot(1,4,1)
            # plt.imshow(img_gray)
            # plt.subplot(1,4,2)
            # plt.imshow(lung_img)
            # plt.subplot(1,4,3)
            # plt.imshow(mask_img)
            # plt.subplot(1,4,4)
            # plt.imshow(img)
            # plt.show()
            # input()
            file_name_ = file_name.replace('/home1/wangzd/194/png/','').replace('.png','')+'_'+str(j)+'.png'
            # print(os.path.join('/home1/wangzd/194/concate_png_bbox',file_name_))
            cv2.imwrite(os.path.join('/ssd2/wangzd/194/concate_png_bbox_2',file_name_),img)
            if "Unidentified" in data['annotations'][j]['type']:
                # label = [0.5,0.5,0]
                continue
            if "Atelectasis" in data['annotations'][j]['type']:
                label = [1,0,1]
            elif data['annotations'][j]['c_score'] != '':
                if "Pneumonia" in data['annotations'][j]['type'] and "Tuberculosis" in data['annotations'][j]['type']:
                    # print(data['annotations'][j]['c_score'])
                    continue
                if "Pneumonia" in data['annotations'][j]['type']:
                    score = int(data['annotations'][j]['c_score'])/5.0
                    if score>=0.8:
                        label=[1,0,0]
                    else:
                        continue
                    #     label=[0.8,0.2,0]
                    # else:
                    #     label = [score,1-score,0]
                elif "Tuberculosis" in data['annotations'][j]['type']:
                    score = float(data['annotations'][j]['c_score'])/5.0
                    if score>=0.8:
                        label=[0,1,0]
                    else:
                        continue
                    #     label=[0.2,0.8,0]
                    # else:
                    #     label = [1-score,score,0]
            elif data['annotations'][j]['c_score'] == '':
                if "Pneumonia" in data['annotations'][j]['type']:
                    label = [1,0,0]
                elif "Tuberculosis" in data['annotations'][j]['type']:
                    label = [0,1,0]
            print(label)
            if 1 in label:
                cate = 'Possitive'
            else:
                cate = 'Negative'
            record = {
                'file_name':os.path.join('/ssd2/wangzd/194/concate_png_bbox_2',file_name_),
                "width": img.shape[1],
                "height": img.shape[0],
                "image_id": data['annotations'][j]['id'],
                "label": label,
                'class': cate,
            }
            dataset_dict.append(record)
with open('/ssd2/wangzd/194/194_bbox_train.json', 'w') as f:
    json.dump(dataset_dict, f)

# import os
# dataset_dict=[]
# image_list = os.listdir("/home1/wangzd/false_positive")

# for i in range(len(image_list)):
#     path=os.path.join("/home1/wangzd/false_positive",image_list[i])
#     img = cv2.imread(path)
#     record={
#         'file_name': path,
#         "width": img.shape[1],
#         "height": img.shape[0],
#         "image_id":10000+i ,
#         "label": [0,0,0],
#         'class': "FalsePositive",
#     }
#     dataset_dict.append(record)
#
# import json
# with open('/home1/wangzd/FP.json', 'w') as f:
#     json.dump(dataset_dict, f)
#             # plt.imshow(img)
#             # plt.show()
#             # bbox = data['annotations'][j]['bbox']
#             # bbox_img = np.zeros_like(img_gray)
#             # bbox_img[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]] = \
#             #     img_gray[bbox[1]: bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
#             # img = np.asarray([bbox_img, lung_img, img_gray])
#             # img = img.transpose([1, 2, 0])




