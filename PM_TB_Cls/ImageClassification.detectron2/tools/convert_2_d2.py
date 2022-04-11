import torch
def convert():
    source_weights = torch.load('/home1/wangzd/Weights/mask_rcnn_swint_T_coco17.pth')['model']
    converted_weights = {}
    keys = list(source_weights.keys())
    # prefix = 'bottom_up.'
    for key in keys:
        converted_weights[key.replace('backbone.','')] = source_weights[key]
    torch.save(converted_weights, '/home1/wangzd/wangzd_chexpert/weights/SwinT_.pth')
convert()

