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


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

PORT = 3200
HOST = "127.0.0.1"

vectorizer = pickle.load(open('models/vectorizer1.pickle', 'rb'))
toxic_classifier = pickle.load(open('models/toxic_linear1_classifier.model', 'rb'))
severe_toxic_classifier = pickle.load(open('models/severe_toxic_linear1_classifier.model', 'rb'))
obscene_classifier = pickle.load(open('models/obscene_linear1_classifier.model', 'rb'))
threat_classifier = pickle.load(open('models/threat_linear1_classifier.model', 'rb'))
insult_classifier = pickle.load(open('models/insult_linear1_classifier.model', 'rb'))
identity_hate_classifier = pickle.load(open('models/identity_hate_linear1_classifier.model', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/textbased_emotion', methods=['GET'])
def predict_textbased_emotion():
    if request.method == 'GET':
        text = request.args.get('text')
        if text:
            text_vector = vectorizer.fit_transform([text])
            toxic_result = toxic_classifier.predict(text_vector)
            severe_toxic_result = severe_toxic_classifier.predict(text_vector)
            obscene_result = obscene_classifier.predict(text_vector)
            threat_result = threat_classifier.predict(text_vector)
            insult_result = insult_classifier.predict(text_vector)
            identity_hate_result = identity_hate_classifier.predict(text_vector)

            its_toxic = toxic_result[0] == 1
            its_severe_toxic = severe_toxic_result[0] == 1
            its_obscene = obscene_result[0] == 1
            its_threat = threat_result[0] == 1
            its_insult = insult_result[0] == 1
            its_identity_hate = identity_hate_result[0] == 1
            inappropriate = its_toxic and its_severe_toxic and its_obscene and its_threat and its_insult and its_identity_hate

            # toxic_result  = "toxic_result"
            # severe_toxic_result  = "severe_toxic_result"
            # obscene_result  = "obscene_result"
            # threat_result  = "threat_result"
            # insult_result  = "insult_result"
            # identity_hate_result  = "identity_hate_result"
            # inappropriate = True

            emotion = "Negative" if inappropriate else "Positive"
            
            countries = ""
            companies = ""
            people = ""
            dates = ""
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