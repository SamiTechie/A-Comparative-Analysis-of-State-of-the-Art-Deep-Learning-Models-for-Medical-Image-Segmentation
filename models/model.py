from . import  sam_model as sam
import cv2
from . import  unet_model as unet
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

class Model:
    def __init__(self, model, dataset):
        self.dataset = dataset
        self.model_name = model
        self.box = True
        self.mask = None
        self.mask_p = None
        self.build()
    def build(self):
        if "SAM" in self.model_name:
            self.model = sam.build_model(self.dataset,self.model_name);
        else:
            self.model = unet.build_model(self.dataset,self.model_name);
    def set_dataset(self,dataset):
        self.dataset = dataset
        self.build()
    def set_model(self, model):
        self.model_name = model
        self.build()
    def predict(self,image, mask = None):
        if "SAM" in self.model_name and "BOX" in self.model_name:
            print("BOX")
            predict, mask =  sam.predict(self.model, image, mask, box = True)
        elif "SAM" in self.model_name:
            print("No BOX")
            predict, mask =  sam.predict(self.model, image, mask, box = False)
        else:
            predict, mask = unet.predict(self.model, image,mask)
        self.mask_p = predict
        self.mask = mask
        return predict
    def evaluation(self):
        mask = self.mask.numpy().squeeze()
        mask = (mask > 0.5).astype(np.uint8).reshape(-1) 
        mask_p = (self.mask_p > 0).astype(np.uint8).reshape(-1)
        iou = jaccard_score( mask,mask_p)
        f1 = f1_score( mask,mask_p)
        accuracy = accuracy_score(mask_p, mask)
        return {"IOU": iou,"F1": f1,"Accuracy": accuracy}

