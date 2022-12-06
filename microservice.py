#!/usr/bin/env python3
"""Machine Learning Challenge

Microservice: Text-based emotion prediction
"""

__author__ = "Carolina Jiménez Moreno <cjimenezm0794@gmail.com>"
__version__ = "1.0.0"

# Standard library imports.
import json
import pickle
# Third party imports.
from flask import Flask, render_template, make_response, jsonify, request
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pycountry

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

PORT = 3200
HOST = "127.0.0.1"

vectorizer = pickle.load(open(f'models/vectorizer2.pickle', 'rb'))
emotion_classifier = pickle.load(open('models/emotion_classifier.model', 'rb'))
# toxic_classifier = pickle.load(open('models/toxic_rbf_classifier.model', 'rb'))
# severe_toxic_classifier = pickle.load(open('models/severe_toxic_rbf_classifier.model', 'rb'))
# obscene_classifier = pickle.load(open('models/obscene_linear1_classifier.model', 'rb'))
# threat_classifier = pickle.load(open('models/threat_linear1_classifier.model', 'rb'))
# insult_classifier = pickle.load(open('models/insult_linear1_classifier.model', 'rb'))
# identity_hate_classifier = pickle.load(open('models/identity_hate_linear1_classifier.model', 'rb'))

def clean_text(text):
    # Lemmatizing the texts
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
    # Remove punctuations, numbers and special characters
    text = re.sub('[^A-Za-z0-9]+', ' ', str(text))
    text = text.strip(' ')
    return text

def get_dates(text):
    # dd/mm/yyyy dd-mm-yyyy 
    search1 = ""
    vals = re.findall(r"[\d]{1,2}[/-][\d]{1,2}[/-][\d]{2,4}", text)
    print(vals)
    if vals:
        search1 = vals
        
    # dd MMM yyyy
    search2 = ""
    vals = re.findall(r"[\d]{1,2} [ADFJMNOS]\w* [\d]{2,4}", text)
    print(vals)
    if vals:
        search2 = vals
        
    search3 = ""
    vals = re.findall(r"([\d]{1,2}\s(January|February|March|April|May|June|July|August|September|October|November|December)\s[\d]{4})", text)
    print(vals)
    if vals:
        search3 = vals
    return search1, search2, search3


def findCountry(stringText):
    for country in pycountry.countries:
        if country.name.lower() == stringText.lower():
            return country.name
    return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/textbased_emotion', methods=['GET'])
def predict_textbased_emotion():
    if request.method == 'GET':
        text = request.args.get('text')
        if text:
            text_cleaned = clean_text(text)
            text_vector = vectorizer.fit_transform([text_cleaned])
            # toxic_result = toxic_classifier.predict(text_vector)
            # severe_toxic_result = severe_toxic_classifier.predict(text_vector)
            # obscene_result = obscene_classifier.predict(text_vector)
            # threat_result = threat_classifier.predict(text_vector)
            # insult_result = insult_classifier.predict(text_vector)
            # identity_hate_result = identity_hate_classifier.predict(text_vector)

            # its_toxic = toxic_result[0] == 1
            # its_severe_toxic = severe_toxic_result[0] == 1
            # its_obscene = obscene_result[0] == 1
            # its_threat = threat_result[0] == 1
            # its_insult = insult_result[0] == 1
            # its_identity_hate = identity_hate_result[0] == 1
            # inappropriate = its_toxic and its_severe_toxic and its_obscene and its_threat and its_insult and its_identity_hate

            toxic_result  = "toxic_result"
            severe_toxic_result  = "severe_toxic_result"
            obscene_result  = "obscene_result"
            threat_result  = "threat_result"
            insult_result  = "insult_result"
            identity_hate_result  = "identity_hate_result"
            inappropriate = True

            emotion = None
            try:
                emotion = emotion_classifier.predict(text_vector)[0]
            except Exception:
                emotion = "negative" if inappropriate else "positive"
                    
            
            countries = findCountry(text)
            companies = ""
            people = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", text)
            dates = get_dates(text)
            hours = ""

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
                    companies=companies,
                    people=people,
                    dates=dates,
                    hours=hours,
                )
        return render_template('error.html', error='Sorry! Unable to parse')

if __name__ == "__main__":
    print(f"Microserver running in port {PORT}")
    app.run(host=HOST, port=PORT)
    print("\nGood")