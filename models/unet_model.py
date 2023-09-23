import cv2
import numpy as np
from . import UnetModel as  models
from PIL import Image
import torch
from torchvision import transforms

def load_model(path, model):
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model
def build_model(dataset, model):
    if model == "UNET":
        model_ = models.unet(3, 1)
    if "SA" in model:
        model_ = models.SAunet(3, 1)
    if "ATT" in model:
        model_ = models.attunet(3, 1)
    model = load_model(f'models/models_checkpoints/{dataset}/{model}.pth',model_)
    return model_
def preprocess(image, mask):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    image = cv2.resize(image, (512, 512)) / 255.0
    mask = cv2.resize(mask, (512, 512))
    convert_tensor = transforms.ToTensor()
    image =  convert_tensor(image).float()
    mask = mask / 255.0
    mask = convert_tensor(mask).float()
    return  torch.unsqueeze(image, dim=0), mask
def predict(model, image_path, mask_path):
    with torch.no_grad():
        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        image, mask = preprocess(image, mask)
        output = model(image)
        result = torch.sigmoid(output)
        threshold = 0.5
        result = (result >= threshold).float()
        prediction = result[0].cpu().numpy()
        mask_p = (prediction * 255).astype('uint8').transpose(1, 2, 0)
        mask_p = np.squeeze(mask_p)
        im = np.uint8(mask_p)
        return im, mask
    
