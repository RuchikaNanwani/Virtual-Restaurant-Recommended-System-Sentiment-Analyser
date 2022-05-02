# Importing the python packages

import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from keras.models import model_from_json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences



# Using pickle to save tokenizer

tokenizer = Tokenizer(oov_token="<OOV>")

# Reading the dataframe in csv format obtained after preprocessing our sample dataset

data_v2 = pd.read_csv('dataset-final-processed.csv', names=["id", "sentiment", "review"])

# Converting review column in str format

Review = data_v2['review'].astype(str)
#(print(np.dtype(Review)))

# Declaring value of oov_token

oov_token = "<OOV>"

# Encoding the labels in categorical values

sentiment_label = data_v2.sentiment.factorize()
#printing the encoded labels
print(sentiment_label)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")

# Fitting reviews in tokenizer

tokenizer.fit_on_texts(Review)



#with open('tokenizer.pickle', 'wb') as handle:
#    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Replacing the words in the sentence with their respective numbers

encoded_docs = tokenizer.texts_to_sequences(Review)
print(encoded_docs)

# The length of every review is different. so, padding it to keep the length common

padded_sequence = pad_sequences(encoded_docs, maxlen=200)

# Checking the available first row of actual and encoded data

print(encoded_docs[0])
print(padded_sequence[0])

# It's time to build our model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.layers import SpatialDropout1D
from tensorflow.keras.layers import Embedding
import tensorflowjs as tfjs


# Declaring the embedding vector length and vocabulary size

embedding_vector_length = 32
vocab_size = len(tokenizer.word_index) + 1
print(tokenizer.word_index)
print(vocab_size)

tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer_json)

# Our model is of type sequential

model = Sequential()



model.add(Embedding(vocab_size, embedding_vector_length, input_length=200))     #adding the embedding layer for our text data
model.add(SpatialDropout1D(0.5))                                                #dropout layer to prevent overfitting
model.add(LSTM(128, dropout=0.5))                                               #LSTM layer
model.add(Dropout(0.2))                                                         #dropout layer to prevent overfitting
model.add(Dense(1, activation='sigmoid'))                                       #adding sigmoid activation function
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy']) #compiling the model
print(model.summary())                                                          #printing the model summary

# Fitting the training data with labels and validation split as 0.2

history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs=10, batch_size=32)


# Serialize model to JSON and save the model

model_json = model.to_json()
with open("BestModel_VRS_Ruchika.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("BestModel_VRS_Ruchika.h5")

tfjs.converters.save_keras_model(model, '/Users/sergiusmiranda/PycharmProjects/pythonProject1')

# Checking the accuracy and validation accuracy

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Checking the loss and validation loss

loss = history.history['loss']
val_loss = history.history['val_loss']

scores = model.evaluate(padded_sequence,sentiment_label[0], verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# Plotting the training data and validation data accuracy visualisation

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and validation accuracy')
plt.show()
plt.figure()

# Plotting the training data and validation data loss visualisation

plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.title('Training and validation loss')
plt.show()
plt.figure()



#with open('tokenizer.pickle', 'rb') as handle:
#    loaded_tokenizer = pickle.load(handle)

#tokenizer = Tokenizer(oov_token="<OOV>")
#test_word = "really good and tasty"
#tokenizer.fit_on_texts(test_word)
#seq = tokenizer.texts_to_sequences(test_word)
#print(seq)
#padded = pad_sequences(seq, 500)
#print(padded)
#prediction = model.predict(padded)
#print(prediction)
#print(round(prediction, 2))
#output = ((np.mean(prediction)).round())
#print(output)
#if output == 1:
#    print("It was a negative Sentiment")
#else:
#    print("It was a positive sentiment")