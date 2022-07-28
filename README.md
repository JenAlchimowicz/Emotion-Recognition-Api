<h1 align="center"> Emotion Recognition Api </h1>
<h3 align="center"> An API that helps you indentify faces and emotions in images and videos </h3>

</br>

<p align="center">
  <img width="500" src=https://user-images.githubusercontent.com/74935134/181221689-3e8ba46e-26bb-46ce-b8fa-75686daa8901.jpg>
</p>

<p align="center">
  <a href="https://www.python.org"> <img src="https://img.shields.io/badge/Python-v3.8-brightgreen"> </a>
  <a href="https://pytorch.org"> <img src="https://img.shields.io/badge/PyTorch-v1.11.0-blue"> </a>
  <a href="https://opencv.org"> <img src="https://img.shields.io/badge/OpenCV-v4.6-blue"> </a>
  <a href="https://fastapi.tiangolo.com"> <img src="https://img.shields.io/badge/FastAPI-v0.79.0-orange"> </a>
  <a href="https://docs.pytest.org/en/7.1.x/contents.html"> <img src="https://img.shields.io/badge/pytest-v7.1.2-orange"> </a>
  <a href="https://docs.pytest.org/en/7.1.x/contents.html"> <img src="https://img.shields.io/github/last-commit/JenAlchimowicz/Emotion-Recognition-Api"> </a>
</p>

## :book: Table of contents
<ol>
  <li><a href="project-description"> ➤ Project description</a></li>
  <li><a href="#folder-structure"> ➤ Folder Structure</a></li>
  <li><a href="#usage"> ➤ Usage</a></li>
    <ul>
      <li><a href="online-api"> Online API</a></li>
      <li><a href="local-usage"> Local usage</a></li>
    </ul>
    <li><a href="#development-process"> ➤ Development process</a></li>
    <ul>
      <li><a href="methods"> Methods </a></li>
      <li><a href="tools"> Tools </a></li>
      <li><a href="datasets"> Datasets </a></li>
      <li><a href="improvement-areas"> Improvement areas </a></li>
    </ul>
    <li><a href="#key-sources"> ➤ References </a></li>
</ol>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## :pencil: Project description

This project identifies faces in an image or video and classifies each face into one of the 7 emotions: ['neutral', 'happy', 'sad', 'surprise', 'fear', 'disgust', 'anger', 'contempt']. For modelling purposes the primary libraries used where PyTorch and OpenCV. For development purposes the primary tools used were pytest, black code formatter, GitHub Actions, FastAPI and Deta.

  - ```Input:``` image or a video
  - ```Output:``` the input with annotated faces [local usage] or a set of coordinates and annotations for each face [API]

**The aim of the project is twofold.** First, emotion recognition technology is used in various real applications. In marketing it is used to better understand the imapct of marketing material during focus group activities. In healthcare it is used to help individuals with autism better identify the emotions and facial expressions they encounter. The automotive industry is experimenting with computer vision technology to monitor the driver's emotional state. An extreme emotional state or drowsiness could trigger an alert for the driver. Therefore, this project serves to show the level of emotion recognition systems today.

Second, the project was developed for learning purposes. Particular emphasis was put on development methods used in the workplace, such as testing, logging, continous integration and developing an API.

</br>

<p align="center">
  <img width="600" src=https://user-images.githubusercontent.com/74935134/181285010-1695aef2-388d-4d69-bcb0-9dafa6166872.png>
</p>

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)


## :cactus: Folder structure
Code structure tree, shows the exact content of the project
```
├── data
│   ├── raw_data
│   ├── train_val_test_split
│   └── predictions
├── requirements.txt
├── architectures
│   ├── ED_model.py
│   └── DAN.py
├── trained_models
│   └── (affecnet8.pth)
├── scripts
│   ├── config.yml
│   ├── etl.py
│   ├── train.py
│   ├── predict.py
│   ├── live_predict.py
│   ├── tests
│   │   └── test_models.py
│   ├── utils
│   │   └── utils.py
│   └── train_utils
│       ├── create_dataloaders.py
│       └── dataloader.py
├── log
│   ├── etl.log
│   ├── predict.log
│   └── results.csv
├── prediction_api
│   ├── api_utility.py
│   └── main.py
├── README.md
└── .github
    └── workflows
        └── ed_app_workflow.yml

```
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## :hammer: Usage

### Online API
Use link: ADD HERE BIG

Example output: ADD IMAGE HERE

### Local usage
1. Git Clone the repo
```
git clone https://github.com/JenAlchimowicz/Emotion-Recognition-Api.git
```

2. Go to project root folder
```
cd Emotion-Recognition-Api
```

3. Setup virtual environment (venv + pip)
```
python -m venv venv
source venv/bin/activate
pip install –r requirements.txt
```

</br>

4. **There are 2 ways you can continue from here. You can:**
  - [**recommended**] Use a pretrained [DAN](https://github.com/yaoing/DAN) model (DAN was trained on [AffectNet](http://mohammadmahoor.com/affectnet/), one of the best emotion recognition datasets available)
  - [**optional**] Train a new model on [fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset and use that to make predictions

</br>
    
  Use a pretrained [DAN](https://github.com/yaoing/DAN) model |  Train a new model
:----------------------------------------------------:|:----------------------------------------------------:
| 1. Download the model weights from [here](https://drive.google.com/file/d/1uHNADViICyJEjJljv747nfvrGu12kjtu/view?usp=sharing) <br/><br/>2. Save in ```trained_models``` directory <br/><br/>3. Modify the ```file_path``` in ```config.yaml``` file to the image/video you want to make predictions on <br/><br/>4. Run ```python scripts/predict.py``` | 1. Run ```python scripts/etl.py``` <br/><br/>2. Run ```python scripts/train.py``` (you can modify number of epochs and other parameters in ```config.yaml```) <br/><br/>3. Modify the ```file_path``` in ```config.yaml``` file to <br/>the image/video you want to make predictions on <br/><br/>4. Run ```python scripts/predict.py``` |

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## :brain: Development process

### :electron: Methods
Emotion recognition is a two-step process:
  1. Detect faces in an image
  2. Classify each face into one of the emotions
  
In this project I relied heavily on pretrained models. For face dection I relied on [Haar Cascade Classifier](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html), and for emotion classification I relied on [DAN](https://github.com/yaoing/DAN) pretrained on [AffectNet](http://mohammadmahoor.com/affectnet/) dataset. I used Haar Cascade because developing a face detection system was not the goal of this project and Haar Cascade provides a stable, easy to implement solution. I used DAN because it is pretrained on an unaccesible to me dataset, and provides better performance than my own implementations on fer2013 dataset.
  
  
### :hammer_and_wrench: Tools
- ```Why PyTorch?``` - most semi-publicly accessible emotion classification datasets are restricted to academics, which mainly use PyTorch. Therefore, I expected to find more good pretrained models available in PyTorch than e.g. Tensorflow
- ```Why config.yaml?``` - in machine learning the data processing, splitting, transforming has as big of an impact on the final result as the choice of model and training parameters. Since I use separate scripts for each of those steps, one would have to remember which arguments were used to run each of the scripts to be able to reproduce results. A config.yaml file puts all arguments together and ensures easy tracking and reproducability.
- ```Why logging?``` - logging leaves an easy to track trace of what was happening during a run. We could print out the results to the command line but in case I want to run e.g. 20 models, the print outputs would quickly become messy. Logging is a clean solution to save all the information in separate files.
- ```Why pytest?``` - testing is crucial for development. Pytest is easy to follow, easy to trace and provides good error reporting.
- ```Why GitHub actions?``` - this is a small project, therefore, quick set-up and simplicity of use is a big advantage. I believed it to be the right tool for the job
- ```Why black?``` - clarity and standardization make it easier for everyone to read code

### :floppy_disk: Datasets
There are a few emotion recognition datasets out there. The three considered in the project were:
- [fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013) [publicly available] - a datset of 35k grayscale 48x48 images annotated with one of 7 emotions. The dataset suffers from a large amount of mislabeled data and the low quality of input images (grayscale and small size).
- [AffectNet](http://mohammadmahoor.com/affectnet/) [available to academics only] - a dataset of 440K RGB images annotated for one of 7 emotions along with the intensity of valence and arousal. State of the art. **Currently unavailable to me**.
- [Real-world Affective Faces](http://www.whdeng.cn/raf/model1.html) - a dataset of 30k RGB images annotated for two of the most prelevant of 7 emotions. **Recently gained access to this dataset**.

### :rocket: Improvement areas
- Experiment with the [Real-world Affective Faces](http://www.whdeng.cn/raf/model1.html) dataset
- Replace Haar Cascade with e.g. YOLO
- Increase test coverage
- Any suggestions are welcome :)

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## :open_book: Key sources
- Structure of the project and key development tools: Geoffrey Hung's articles [here](https://towardsdatascience.com/from-jupyter-notebook-to-sc-582978d3c0c) and [here](https://towardsdatascience.com/from-scripts-to-prediction-api-2372c95fb7c7)
- DAN paper by Wen, Zhengyao and Lin, Wenzhong and Wang, Tao and Xu, Ge: https://arxiv.org/pdf/2109.07270.pdf
- Testing webinar: https://www.youtube.com/watch?v=ytI4Xapvx1w
- Haar implemetation: https://www.youtube.com/watch?v=7IFhsbfby9s
- Readme design: https://github.com/ma-shamshiri/Human-Activity-Recognition#readme
- FastAPI tutorial: https://github.com/aniketmaurya/tensorflow-fastapi-starter-pack


![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
