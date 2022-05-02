from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras import layers
from keras import regularizers
embedding_layer = Embedding(1000, 64)
import keras


from keras.callbacks import ModelCheckpoint


data_v2 = pd.read_csv('dataset-final-processed.csv', names=["id", "sentiment", "review"])

tokenizer = Tokenizer(oov_token="<OOV>")
es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

split = round(len(data_v2)*0.7)

train_reviews = data_v2['Comment'][:split]
train_label = data_v2['Label'][:split]

test_reviews = data_v2['Comment'][split:]
test_label = data_v2['Label'][split:]


training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

for row in train_reviews:
    training_sentences.append(str(row))
for row in train_label:
    training_labels.append(row)
for row in test_reviews:
    testing_sentences.append(str(row))
for row in test_label:
    testing_labels.append(row)

vocab_size = 40000
embedding_dim = 16
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
print(word_index)

sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)
testing_sentences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sentences, maxlen=max_length)


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Conv1D(128, 5, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Conv1D(128, 5, activation='linear'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.GlobalMaxPooling1D(),
    #tf.keras.layers.LSTM(128, dropout=0.6),
    #tf.keras.layers.Bidirectional(layers.LSTM(128)),
    #tf.keras.layers.Dense(32, kernel_regularizer=regularizers.l1(0.002), activation='softmax'),
    #tf.keras.layers.Dropout(0.6),
    #tf.keras.layers.Dense(25, kernel_regularizer=regularizers.l1(0.002), activation='relu'),
    #tf.keras.layers.Dropout(0.6),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model = Sequential()
#model.add(layers.Embedding(vocab_size, 40, input_length=max_length))
#model.add(layers.Bidirectional(layers.LSTM(128)))
#model.add(layers.Dense(250, activation='relu')),
#model.add(keras.layers.Dropout(0.4)),
#model.add(layers.Dense(150, activation='relu')),
#model.add(keras.layers.Dropout(0.4)),
#model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
checkpoint2 = ModelCheckpoint("best_model2.h5", monitor='val_accuracy', verbose=1,save_best_only=True, mode='auto', save_freq=1, save_weights_only=False)

model.summary()

training_labels_final = []
testing_labels_final = []

training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

num_epochs = 40
#history = model.fit(padded, training_labels_final, epochs=25, validation_data=(testing_padded, testing_labels_final), callbacks=[checkpoint2])
history = model.fit(padded, training_labels_final, epochs=6, validation_data=(testing_padded, testing_labels_final), callbacks=[es_callback])
model.save('restaurant_reviews.h5')

import pickle

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

#Visualisation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs=range(len(acc))

plt.plot(epochs, acc, 'r', 'Training Accuracy')
plt.plot(epochs, val_acc, 'b', 'Validation Accuracy')
plt.title('Training and validation accuracy')
plt.show()
plt.plot(epochs, loss, 'r', 'Training Loss')
plt.plot(epochs, val_loss, 'b', 'Validation Loss')
plt.title('Training and validation loss')
plt.show()

loaded_model = tf.keras.models.load_model('restaurant_reviews.h5')

with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

    txt = testing_sentences[10]
    seq = loaded_tokenizer.texts_to_sequences(['Wow loved the place'])
    padded = pad_sequences(seq, maxlen=max_length)
    pred = (model.predict_step(padded))
    print(pred)

  #  txt1 = testing_sentences[10]
  #  seq1 = loaded_tokenizer.texts_to_sequences(['Wow loved the place'])
  #  padded1 = pad_sequences(seq1, maxlen=max_length)
  #  pred1 = (model.predict_step(padded1))
  #  print(pred1)

    if pred > 0.30:
        print("The sentiment was positive")
    elif pred < 0.30:
        print("The sentiment was negative")

    #if pred1 > 0.30:
    #    print("The sentiment was positive")
   # elif pred1 < 0.30:
     #   print("The sentiment was negative")
