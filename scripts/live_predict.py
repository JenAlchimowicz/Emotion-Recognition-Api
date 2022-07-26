import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import os

import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from architectures.DAN import DAN

'''
Script that opens the computer camera and makes live predictions.

Aguments:
    - model_path [str] - file path to a trained DAN model (e.g. 'trained_models/affecnet8.pth')

Outputs:
    - Live prediction displayed on the screen. Does not save videos, to save a predicted clip first record a video and then use the more developed script 'predict.py'
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mp', '--model_path', type=str, default='trained_models/affecnet8.pth', 
                         help='Path to the DAN pretrained model')
    return parser.parse_args()


class Model_pred:
    def __init__(self, MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.labels = ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
        self.model = DAN(num_head=4, num_class=8, pretrained=False)
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    def detect_faces(self, img_:np.array):
        faces = self.face_cascade.detectMultiScale(img_, 1.2, 6)
        return faces
    
    def predict_emotion(self, img_cv:np.array):
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))    
        faces = self.detect_faces(img_cv)

        for (x,y,w,h) in faces:
            cropped_face = img_pil.crop((x,y, x+w, y+h))
            cropped_face = self.data_transforms(cropped_face).unsqueeze(0)

            with torch.set_grad_enabled(False):
                output = self.model(cropped_face)
                pred = self.labels[int(torch.argmax(output))]
            
            # Add rectangels
            img_cv = cv2.rectangle(img_cv, (x,y), (x+w,y+h), (0,255,0), 2)
            (w, h), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            img_cv = cv2.rectangle(img_cv, (x, y-20), (x+w, y), (0,255,0), -1)
            img_cv = cv2.putText(img_cv, pred, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

        return img_cv

        
if __name__ == '__main__':
    ######################
    # Extract arguments from parser
    ######################
    args = parse_args()
    MODEL_PATH = args.model_path

    ######################
    # Initialise model
    ######################
    model = Model_pred(MODEL_PATH)

    ######################
    # Open camera and make predictions
    ######################  
    capture = cv2.VideoCapture(0)
    while True:
        ret, img = capture.read()
        if ret == True:
            output_img = model.predict_emotion(img)
            cv2.imshow('Live emotion prediction', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    capture.release()
    cv2.destroyAllWindows()