
# Airline Sentiment Analysis API <a name="TOP"></a>

- - - - 
## Airline Sentiment Analysis

This repository provides an API endpoint that can accept a text and return associated sentiment with it  
For `Swagger Documentation` go to [API](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/api.json)

## Pipeline

![Screenshot](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/imgs/Pipeline.png)


## Installation

Install Project and Create Environment

```bash
git clone https://github.com/TashvikDhamija/Airline-Sentiment-Analysis
cd Airline-Sentiment-Analysis
conda env create --name airlinesentiment --file requirements.txt
conda activate airlinesentiment
```

To run API

```bash
uvicorn api:app --reload
```

Click [here](http://127.0.0.1:8000/) to access API \
Click [here](http://127.0.0.1:8000/docs) to access Swagger UI of API

## Model Training

To train model

```bash
python main.py [-h] [--model_name MODEL_NAME] [--epochs EPOCHS] [--lr LR]
               [--batch_size BATCH_SIZE] [--features FEATURES] [--seed SEED]
               [--log_interval LOG_INTERVAL] [--test_interval TEST_INTERVAL]
               [--save SAVE]
```

To make changes to the classifier, go to [model.py](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/model.py) \
To make changes to the data loading, go to [dataset.py](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/dataset.py) \
To make changes to the preprocessing, go to [preprocessing.py](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/preprocessing.py) and [dataset.py](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/dataset.py) 

Currently, the following classifier is being used: \
![Screenshot](https://github.com/TashvikDhamija/Airline-Sentiment-Analysis/blob/master/imgs/Classifier.png)

## Model Performance


| Model | Accuracy     | Speed |  Size             | 
| :-------- | :------- | :------------------------- |:--|
| `paraphrase-MiniLM-L3-v2` | 91.7533% |19000 | 61MB|
| `all-MiniLM-L6-v2` | 91.4414% |14200 | 80MB|
| `all-distilroberta-v1` | 92.2037% |4000 | 290MB|
| `all-mpnet-base-v2` | 91.9958% |2800 | 420MB|

