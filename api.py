from fastapi import FastAPI
from main import Trainer
from preprocessing import process

class Args:
    def __init__(self):
        self.features = 384
        self.model_name = 'sentence-transformers/paraphrase-MiniLM-L3-v2'

PATH = '_best.pt'

app = FastAPI()

model = Trainer(Args(), pred=True)
model.load(PATH)
model.classifier.eval()

@app.get('/')
def main():
    return {'message': 'Hello! Welcome to airline tweets sentiment analysis!'}

@app.post('/query_sentiment_and_score/')
def get_sentiment_and_score(text:str):
    output = model([text]).item()
    sentiment = 'positive' if round(output) == 1 else 'negative'
    return {"input sentence": text, "positive_sentiment": output, "negative_sentiment": 1-output, 'sentiment': sentiment}

@app.post('/query_sentiment/')
def get_sentiment(text:str):
    output = model([text]).item()
    sentiment = 'positive' if round(output) == 1 else 'negative'
    return {"input sentence": text, 'sentiment': sentiment}

@app.post('/query_preprocessing/')
def get_preprocessed_sentence(text:str):
    output = process(text)
    return {"input sentence": text, 'preprocessed sentence': output}