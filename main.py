
#::: Import modules and packages :::
# Flask utils
from flask import Flask, request,  render_template
import sys


# Import Kerasapp
app = Flask(__name__)

# ::: Prepare Keras Model :::
# Model files
Token = 'tokenizer.json'
MODEL_ARCHITECTURE = 'BestModel_VRS_Ruchika.json'
MODEL_WEIGHTS = 'BestModel_VRS_Ruchika.h5'
#with open('tokenizer.pickle', 'rb') as handle:
#    tokenizer = pickle.load(handle)
#dependencies
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json

#import kafka consumer

#from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

#::: Flask App Engine :::
# Define a Flask
json_file2 = open(Token)
loaded_token_json = json_file2.read()
json_file2.close()
tokenizer = tokenizer_from_json(loaded_token_json)


# Load the model from external files
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Get weights into the model
model.load_weights(MODEL_WEIGHTS)
print('Model loaded. Check http://127.0.0.1:5000/')

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    global p
    if request.method == 'POST':
        str_features = [x for x in request.form.values()]
        tokenizer.fit_on_texts(str_features)
        print(tokenizer.word_index)
        seq = tokenizer.texts_to_sequences(str_features)
        print(seq)
        #seq = [x for x in seq if x != []]
        padded = pad_sequences(seq, 200)
        print(padded)
        prediction = (model.predict(padded))
        print(prediction)
        op = prediction
        #output1 = op[len(op) - 1]
        #output2 = op[len(op) - 2]
        #output = output1 + output2
        output = (np.mean(op))
        print(output)
        output = output.round()
        if output == 0:
            p = "It was a positive Sentiment"
            print(p)
        elif output == 1:
            p = "It was a negative sentiment"
            print(p)
        output = 0
        prediction = []
        op = []
        seq = []
        padded = []
    return render_template('home.html', prediction="Sentiment is :) or :( "+p)

if __name__ == '__main__':
    app.run(debug = True)