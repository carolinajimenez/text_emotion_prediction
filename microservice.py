#!/usr/bin/env python3
"""Machine Learning Challenge

Microservice: Text-based emotion prediction
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import re
import json
import pickle
# Third party imports.
import nltk
from nltk.stem import WordNetLemmatizer
import pycountry
from flask import Flask, render_template, make_response, jsonify, request
import torch
import torch.nn as nn


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# APP CONFIGURATIONS

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

PORT = 3200
HOST = "127.0.0.1"


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# INITIALIZE MODELS

vectorizer = pickle.load(open(f'models/vectorizer2.pickle', 'rb'))
emotion_classifier = pickle.load(open('models/emotion_classifier.model', 'rb'))

with open('models/word_dict.json') as json_file:
    word_dict = json.load(json_file)
labels_wigths = 'models/model_26_87.12.pth'
# 10-column embed model
embedding = nn.Embedding(len(word_dict), 10)



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# FUNCTIONS

# Base model: network of a neuron with 6 outputs
class base_line(nn.Module):
  def __init__(self,fin,out):
    super(base_line,self).__init__()
    self.out = out
    self.fin = fin
    self.fc1 = nn.Linear(self.fin,2048)
    self.fc2 = nn.Linear(2048,1024)
    self.fc3 = nn.Linear(1024,512)
    self.relu = nn.ReLU()
    self.fc4 = nn.Linear(512,self.out)
    self.sigmoid = nn.Sigmoid()

  def forward(self,x):
    out = self.fc1(x)
    out = self.fc2(out)
    out = self.fc3(out)
    our = self.relu(out)
    out = self.fc4(out)
    out = self.sigmoid(out)
    return out

model = base_line(10,6)
model.load_state_dict(torch.load(labels_wigths, map_location=torch.device('cpu')))
model.eval()

def average_tensor(x):
    """
    Function that calculates average vector
    """
    tensor_d = torch.zeros((1,10))
    for t in x:
        tensor_d += t
    return tensor_d/x.shape[0]

def coded_words(x, word_dict):
    return [word_dict[w] for w in x if w in word_dict]

def converter(words):
    return average_tensor( embedding(torch.tensor(coded_words(words.split(), word_dict))))

nltk.download('omw-1.4')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
def clean_text(text):
    """
    Process text - Lemmatizing the texts
    """
    # removing aphostrophe words
    text = text.lower()
    text = re.sub(r"what's", "what is ",str(text)) 
    text = re.sub(r"'s", " ", str(text)) 
    text = re.sub(r"'ve", " have ", str(text)) 
    text = re.sub(r"can't", "cannot ", str(text)) 
    text = re.sub(r"ain't", 'is not', str(text)) 
    text = re.sub(r"won't", 'will not', str(text)) 
    text = re.sub(r"n't", " not ", str(text)) 
    text = re.sub(r"i'm", "i am ", str(text)) 
    text = re.sub(r"'re", " are ", str(text)) 
    text = re.sub(r"'d", " would ", str(text)) 
    text = re.sub(r"'ll", " will ", str(text)) 
    text = re.sub(r"'scuse", " excuse ", str(text)) 
    text = re.sub('W', ' ', str(text)) 
    text = re.sub(' +', ' ', str(text))
    # Remove hyperlinks
    text = re.sub(r"https?://\S+|www\.\S+", ' ', str(text))
    # Remove punctuations, numbers and special characters
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    text = lemmatizer.lemmatize(text)
    text = text.strip(' ')
    return text

def get_dates(text):
    """
    Get the dates that are in text
    """
    # dd/mm/yyyy; dd-mm-yyyy; dd mm yyyy 
    search = ""
    vals = re.findall(r"[\d]{1,2}[\s/-][\d]{1,2}[/-][\d]{2,4}", text)
    if vals:
        for i in vals:
            if search != "":
                search += ", "
            search += f"{i}"
        
    # dd MMM yyyy
    vals = re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{2,4}", text)
    if vals:
        for i in vals:
            if search != "":
                search += ", "
            search += f"{i}"
    
    # Mar; March
    vals = re.findall(r'/^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)$/i', text)
    print(vals)
    if vals:
        for i in vals:
            if search != "":
                search += ", "
            search += f"{i}"

    # Mar-20-2009; Mar 20, 2009; March 20, 2009; Mar. 20, 2009; Mar 20 2009;
    vals = re.findall(r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z.,-]*[\s-]?(\d{1,2})?[,\s-]?[\s]?\d{4}', text, re.I|re.M)
    if vals:
        for i in vals:
            if search != "":
                search += ", "
            search += f"{i}"

    return "-" if search == "" else search

def find_country(text):
    """
    Get the countries that are in text
    """
    countries = ""
    for country in pycountry.countries:
        if country.name in text:
            if countries != "":
                countries += ", "
            countries += country.name
    return "-" if countries == "" else countries

def get_people_names(text):
    """
    Get the people names that are in text
    """
    names = ""
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    for tupla in tagged:
        if tupla[1] == "NNP":
            if names != "":
                names += ", "
            names += tupla[0]
    return names



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# RUN APP

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/textbased_emotion', methods=['GET'])
def predict_textbased_emotion():
    if request.method == 'GET':
        text = request.args.get('text')
        if text:
            text_cleaned = clean_text(text)
            out = converter(text)
            res = model(out)

            toxic_result = "Yes" if res[0][0].item() > 0.29 else "No"
            severe_toxic_result = "Yes" if res[0][1].item() > 0.29 else "No"
            obscene_result = "Yes" if res[0][2].item() > 0.29 else "No"
            threat_result = "Yes" if res[0][3].item() > 0.29 else "No"
            insult_result = "Yes" if res[0][4].item() > 0.29 else "No"
            identity_hate_result = "Yes" if res[0][5].item() > 0.29 else "No"

            its_toxic = toxic_result == "Yes"
            its_severe_toxic = severe_toxic_result == "Yes"
            its_obscene = obscene_result == "Yes"
            its_threat = threat_result == "Yes"
            its_insult = insult_result == "Yes"
            its_identity_hate = identity_hate_result == "Yes"
            inappropriate = its_toxic and its_severe_toxic and its_obscene and its_threat and its_insult and its_identity_hate

            text_vector = vectorizer.fit_transform([text_cleaned])
            emotion = None
            try:
                emotion = emotion_classifier.predict(text_vector)[0]
                if emotion == "negative" and not inappropriate:
                    emotion = "Positive"
            except Exception:
                print("emotion: .-.")
                emotion = "Negative" if inappropriate else "Positive"

            countries = find_country(text)
            paises = countries.split(', ')
            
            #names = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", text)
            names = get_people_names(text)
            for p in paises:
                names = names.replace(p, "")
            
            dates = get_dates(text)
            fechas = dates.split(', ')
            for f in fechas:
                names = names.replace(f, "")
            people = names
            

            #people = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", text)

            hours = re.findall(r'([\d\s\w:]+) - ([\d\s\w:]+)', text)

            #return make_response(jsonify({'sentiment': result[0], 'text': text, 'status_code':200}), 200)
            return render_template(
                    'index.html',
                    text=text,
                    inappropriate=inappropriate,
                    toxic_result=toxic_result,
                    severe_toxic_result=severe_toxic_result,
                    obscene_result=obscene_result,
                    threat_result=threat_result,
                    insult_result=insult_result,
                    identity_hate_result=identity_hate_result,
                    emotion=emotion,
                    countries=countries,
                    people=people,
                    dates=dates,
                    hours=hours,
                )
        return render_template('error.html', error='Sorry! Unable to parse')

if __name__ == "__main__":
    print(f"Microserver running in port {PORT}")
    app.run(host=HOST, port=PORT)
    print("\n\nGood bye!")