# Importing the packages for setting up kafka producer - consumer pipeline
# Importing the packages for predicting the review received from the pipeline

from kafka import KafkaConsumer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import numpy as np
import sys

# Model files
Token = 'tokenizer.json'
MODEL_ARCHITECTURE = 'BestModel_VRS_Ruchika.json'
MODEL_WEIGHTS = 'BestModel_VRS_Ruchika.h5'

json_file2 = open(Token)
loaded_token_json = json_file2.read()
json_file2.close()
tokenizer = tokenizer_from_json(loaded_token_json)

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
print('Model loaded')

bootstrap_servers = ['localhost:9092']
topicName = 'json_kafka'
consumer = KafkaConsumer(topicName,bootstrap_servers=bootstrap_servers, auto_offset_reset='latest')

for message in consumer:
        global p
        #print ("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,message.offset, message.key,message.value))
        message = str(message.value, encoding='utf-8')
        data = [message]
        tokenizer.fit_on_texts(data)
        #print(tokenizer.word_index)
        seq = tokenizer.texts_to_sequences(data)
        print(seq)
        # seq = [x for x in seq if x != []]
        padded = pad_sequences(seq, 200)
        print(padded)
        prediction = (model.predict(padded))
        print(prediction)
        op = prediction
        # output1 = op[len(op) - 1]
        # output2 = op[len(op) - 2]
        # output = output1 + output2
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
        sys.exit()