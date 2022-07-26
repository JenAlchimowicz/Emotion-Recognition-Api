import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from architectures.DAN import DAN

import torch
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
from io import BytesIO


def read_img(image_encoded):
    '''
    Reads an encoded image file from API call and outputs a PIL image (in RGB)
    '''
    try:
        img_pil = Image.open(BytesIO(image_encoded)).convert('RGB')
    except: 
        raise TypeError('Only image formats supported by PIL are allowed')
    return img_pil


class Predictor:
    def __init__(self, model_path:str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        self.data_transforms = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    def detect_faces(self, img_):
        img_ = cv2.cvtColor(np.asarray(img_),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img_, 1.2, 6)
        return faces

    def predict(self, img_pil:Image.Image):
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        faces = self.detect_faces(img_pil)
        response = []

        for (x,y,w,h) in faces:
            cropped_face = img_pil.crop((x,y, x+w, y+h))
            cropped_face = self.data_transforms(cropped_face).unsqueeze(0)
            
            with torch.set_grad_enabled(False):
                output = self.model(cropped_face)
                pred = self.labels[int(torch.argmax(output))]
                response.append({'x':int(x), 'y':int(y), 'width':int(w), 'height':int(h), 'label':str(pred)})
        return response



