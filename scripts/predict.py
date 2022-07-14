import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import argparse
import filetype
import os


import sys
import os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import parse_config, configure_logger
from architectures.my_model import ED_model
from architectures.DAN import DAN

'''
Script that takes a trained model and an image/video, detects faces in the image/video, for each detected face predicts an emotion and shows/saves the result

Aguments:
    - file_path [str] - path to the file to make prediction on
    - model_type [str] - either 'affectnet8' or 'ED'
    - model_path [str] - file path to a trained model (e.g. 'trained_models/affecnet8.pth'). Parameters must be consistent with the model type selected in model_type
    - export_path [str] - folder path where to save the results (e.g. 'data/predictions/')
    - export_file_name [str] - name of the file to save (without extension, e.g. 'prediction0')
    - save_img [bool] - whether to save new image (with bounding boxes and labels overlayed on top of the original image) in the export_path directory
    - show_result [bool] - whether to display the new image on the screen after prediction

Outputs:
    - New image [jpg] saved in the export_path directory (if save_img == True)
    - Labels [csv] saved in the export_path directory (if save_pred == True)
    - Image [jpg] displayed on the screen (if show_results == True)
'''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfp', '--config_file_path', type=str, default='./scripts/config.yml', 
                         help='Path to the YAML configuration file')
    return parser.parse_args()



class Model_pred():
    def __init__(self, MODEL_TYPE, MODEL_PATH):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if MODEL_TYPE == 'affectnet8':
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

        elif MODEL_TYPE == 'ED':
            self.data_transforms = torch.nn.Sequential(
                    transforms.Resize((48,48)),
                    transforms.ToTensor(),
                    transforms.Grayscale(num_output_channels=1))
            self.labels = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', "surprise"]
            self.model = ED_model(in_channels=1, out_channels=7).to(self.device)
            self.model.load_state_dict(torch.load(MODEL_PATH))

        else:
            raise ValueError('Incorrect model type entered, options are: "affectnet8" and "ED"')

        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')


    def detect_faces(self, img_):
        img_ = cv2.cvtColor(np.asarray(img_),cv2.COLOR_RGB2BGR)
        faces = self.face_cascade.detectMultiScale(img_, 1.2, 6)
        return faces
    
    def predict_emotion(self, path, file_type):

        if file_type == 'image':
            img0 = Image.open(path).convert('RGB')
            img_cv = np.array(img0) 
            img_cv = img_cv[:, :, ::-1].copy() 
            
            faces = self.detect_faces(img0)
            preds = []

            for (x,y,w,h) in faces:
                cropped_face = img0.crop((x,y, x+w, y+h))
                cropped_face = self.data_transforms(cropped_face).unsqueeze(0)

                with torch.set_grad_enabled(False):
                    output = self.model(cropped_face)
                    pred = self.labels[int(torch.argmax(output))]
                    preds.append(pred)
            
                # Add rectangels
                img_cv = cv2.rectangle(img_cv, (x,y), (x+w,y+h), (0,255,0), 2)
                (w, h), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                img_cv = cv2.rectangle(img_cv, (x, y-20), (x+w, y), (0,255,0), -1)
                img_cv = cv2.putText(img_cv, pred, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

            return img_cv, preds
        
        
        elif file_type == 'video':
            cap = cv2.VideoCapture(path)
            output_array = []

            while True:
                ret, img_cv = cap.read()
                preds = []
                
                if ret == True:
                    img_pil = Image.fromarray(img_cv)
                    faces = self.detect_faces(img_cv)
                    for (x,y,w,h) in faces:
                        cropped_face = img_pil.crop((x,y, x+w, y+h))
                        cropped_face = self.data_transforms(cropped_face).unsqueeze(0)

                        with torch.set_grad_enabled(False):
                            output = self.model(cropped_face)
                            pred = self.labels[int(torch.argmax(output))]
                            preds.append(pred)

                        # Add rectangels
                        img_cv = cv2.rectangle(img_cv, (x,y), (x+w,y+h), (0,255,0), 2)
                        (w, h), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                        img_cv = cv2.rectangle(img_cv, (x, y-20), (x+w, y), (0,255,0), -1)
                        img_cv = cv2.putText(img_cv, pred, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
                    
                    output_array.append(img_cv)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                else: break

            cap.release()
            return output_array, preds

        

        
if __name__ == '__main__':
    ######################
    # Configure logger
    ######################
    logger = configure_logger(__file__, 'log/predict.log')

    ######################
    # Extract arguments from parser
    ######################
    args = parse_args()
    config = parse_config(args.config_file_path)
    FILE_PATH = config['predict']['file_path']
    MOEDL_TYPE = config['predict']['model_type']
    MODEL_PATH = config['predict']['model_path']
    EXPORT_PATH = config['predict']['export_path']
    EXPORT_FILE_NAME = config['predict']['export_file_name']
    SAVE_FILE = config['predict']['save_file']
    SHOW_RESULT = config['predict']['show_result']

    if SAVE_FILE + SHOW_RESULT < 1:
        print('WARNING, user did not specify to either show or save results, so nothing happens. Please modify the "SAVE_FILE" or "SHOW_RESULTS" arguments in the configuration file')

    ######################
    # Make predictions
    ######################
    logger.info('---------------- Start prediction script ----------------')
    model = Model_pred(MOEDL_TYPE, MODEL_PATH)
    logger.info('Model created')

    file = filetype.guess(FILE_PATH)
    file_type = file.mime.split('/')[0]
    assert (file_type=='video' or file_type=='image'), 'The only accepted formats are images and videos'
    logger.info(f'File type detected as: {file_type}')

    out, labels = model.predict_emotion(FILE_PATH, file_type)
    logger.info('Predictions made')

    ######################
    # Save/show predictions
    ######################
    if file_type == 'image':
        if SHOW_RESULT:
            cv2.imshow('pred_img', out)
            cv2.waitKey()
        if SAVE_FILE:
            cv2.imwrite(EXPORT_PATH+EXPORT_FILE_NAME+'.jpg', out)
            logger.info(f'Prediction image saved in {EXPORT_PATH+EXPORT_FILE_NAME+".jpg"}')
    if file_type == 'video':
        height, width, channels = out[0].shape
        writer = cv2.VideoWriter(EXPORT_PATH+EXPORT_FILE_NAME+'.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20, (width, height))
        for frame in out:
            writer.write(frame)
            if SHOW_RESULT:
                cv2.imshow('video_prediction', frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    break
        writer.release()
        if not SAVE_FILE:
            os.remove(EXPORT_PATH+EXPORT_FILE_NAME+'.avi')
