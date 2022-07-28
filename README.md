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
      <li><a href="tools"> Tools </a></li>
      <li><a href="datasets"> Datasets </a></li>
      <li><a href="improvement-areas"> Improvement areas </a></li>
    </ul>
    <li><a href="#references"> ➤ References </a></li>
    <li><a href="#one-more-for-looks"> ➤ Something else </a></li>
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

# Usage

  ## Online API
  just use link

  ## Local usage
  
 
# Development process
## Tools
## Datasets
## Improvement areas

# References
# One more for looks

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)
