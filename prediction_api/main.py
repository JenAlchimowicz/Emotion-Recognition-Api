from unicodedata import name
from fastapi import FastAPI
import uvicorn
from starlette.responses import RedirectResponse

from fastapi import File, UploadFile
from api_utility import read_img, Predictor

#######################
# Initialisation
#######################
app_desc = """<h2>Try this API by uploading an image </h2>
<h2> Instructions: `POST/predict` -> `Try it out` -> `Choose File` -> `Execute`</h2>
<br>by Jen Alchimowicz"""

app = FastAPI(title='Emotion Recognition API', description=app_desc)
model = Predictor(model_path="trained_models/affecnet8.pth")

#######################
# Prediction API
#######################
@app.get('/', include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.post('/predict')
async def predict_img(file: UploadFile=File()):
    img = read_img(await file.read())
    preds = model.predict(img)
    return preds


if __name__ == '__main__':
    uvicorn.run(app)