from unicodedata import name
from fastapi import FastAPI
import uvicorn

from fastapi import File, UploadFile
from api_utility import read_img, Predictor

#######################
# Initialisation
#######################
app = FastAPI()
model = Predictor(model_path="trained_models/affecnet8.pth")

#######################
# Prediction API
#######################
@app.get('/')
async def root():
    return {"message": "this is root, prediction endpoint should be `predict`!, update endpoint should be `update_model`"}

@app.post('/predict')
async def predict_img(file: UploadFile=File()):
    img = read_img(await file.read())
    preds = model.predict(img)
    return preds


if __name__ == '__main__':
    uvicorn.run(app)