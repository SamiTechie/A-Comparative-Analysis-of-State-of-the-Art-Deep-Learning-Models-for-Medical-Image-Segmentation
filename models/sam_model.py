from .SamModel import  SamModel
from box import Box
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torch
from transformers import SamProcessor
import numpy as np
import cv2
import albumentations as A
class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms
        

    def __call__(self, img, mask, *args, **kwargs):
        return self.transforms(image=np.array(img), mask = np.array(mask))
def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))
        
        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 256, 256] # if there is no mask in the array, set bbox to image size
def config(model):
  config = {
        "batch_size": 1,
    "num_epochs": 10,
    "opt": {
        "learning_rate": 1e-5,
        "weight_decay": 1e-9,
    },
    "model": {
        "type": 'vit_b',
        "checkpoint": model,
        "freeze": {
            "image_encoder": False,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    }
    }
  return Box(config)

def build_model(dataset, model):
    model = SamModel(config(f'models/models_checkpoints/{dataset}/{model}.pth'))
    model.setup()
    model.eval() 
    return model

def preprocess(image, mask):
    transform = Transforms(A.Compose([
        A.Resize(1024, 1024),
        ToTensorV2()
    ]))
    example = transform(image,mask)
    image =  example['image'].float()
    mask =  example['mask']
    prompt = get_bounding_box(mask.unsqueeze(0).numpy()[0])
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    input_  = processor(image, input_boxes=[[prompt]], return_tensors="pt")
    return input_['pixel_values'],input_['input_boxes'], mask

def predict(model, image_path, mask_path):
    with torch.no_grad():
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        image, boxes,mask = preprocess(image,mask)
        mask_p = model(image, boxes)
        mask_p= torch.sigmoid(mask_p[0][0].unsqueeze(0)).detach().numpy().squeeze()
        mask_p = (mask_p > 0.5).astype(np.uint8) * 255
        im = Image.fromarray(np.uint8(mask_p))
        return mask_p, mask
