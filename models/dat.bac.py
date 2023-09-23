from torch.utils.data import Dataset
import glob
import os
from albumentations.pytorch import ToTensorV2
import torch
import imageio
from PIL import Image
from transformers import SamProcessor
from scipy import ndimage
from torchvision.io import read_image
import cv2
import imageio
import numpy as np
import albumentations as A
from torchvision.transforms import transforms
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
class BasicDataset(Dataset):
    def __init__(self, images_dir,masks_dir,processor, transform=None):
        self.images = sorted(glob.glob(images_dir+ "/*png"))
        self.masks = sorted(glob.glob(masks_dir+ "/*png"))
        self.processor = processor
        self.transform = Transforms(transform)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, indx):
        # Apply transformations if specified
        image = cv2.imread(self.images[indx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(self.masks[indx], cv2.IMREAD_GRAYSCALE)
        #mask = imageio.imread(self.masks[indx])
        mask = cv2.imread(self.masks[indx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            example = self.transform(image,mask)
            image = example['image'].float()
            mask  = example['mask']
            prompt = get_bounding_box(mask.unsqueeze(0).numpy()[0])
        mask = mask.unsqueeze(0)
        mask[mask > 0] = 1
        #mask = transforms.Resize((256,256))(mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        # remove batch dimension which the processor adds by default
        #inputs["mask"] = transforms.Resize((256,256))(mask)
        inputs["mask"] = mask
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        return inputs
