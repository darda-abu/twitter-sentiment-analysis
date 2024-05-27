from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
from helper import *
import os

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

if not os.path.exists("cardiffnlp"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    model.save_pretrained(MODEL)
    tokenizer.save_pretrained(MODEL)
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL,from_local=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

labels = label_mapping(f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt")


def predict(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    output = ""
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        output+=(f"{i+1}) {l} {np.round(float(s), 4)}\n")
    return output

text = open('input.txt').read()
    
print(predict(text))