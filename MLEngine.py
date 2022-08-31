import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import os
import numpy as np

class MLEngine:
    def __init__(self,modelPath= "C:\\Users\\thoma\\Desktop\\IOT_FINAL\\ML\\WI_MODEL\\model_weights.pth"):
        self.model = models.vgg16(pretrained = True)
        self.model.classifier = nn.Sequential(
        nn.Linear(25088, 4096, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(4096, 2048, bias = True),
        nn.ReLU(inplace = True),
        nn.Dropout(0.4),
        nn.Linear(2048, 5)
        )
        self.tensorProcess = T.Compose([
        T.ToTensor(),
        ])
        self.device=torch.device('cpu')
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(modelPath))
        DIR_TRAIN = "C:\\Users\\thoma\\Desktop\\IOT_FINAL\\ML\\WI_MODEL\\train\\"
        self.classes = os.listdir(DIR_TRAIN)
    def prepareImage(self, image):
        outimage = cv2.imread(image, cv2.IMREAD_COLOR)
        outimage = cv2.cvtColor(outimage, cv2.COLOR_BGR2RGB).astype(np.float32)
        outimage /= 255.0
        return outimage
    def inferPhoto(self,image_in):
        image=self.prepareImage(image_in)
        #Send to CPU
        image = self.tensorProcess(image).unsqueeze(0)
        result = self.model(image)
        _, predicted = torch.max(result, 1)
        return self.classes[predicted[0]]