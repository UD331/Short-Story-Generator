import tensorflow as tf
import numpy as np
import os
import keras
from keras.preprocessing import sequence
from google.colab import files

path_to_file = list(files.upload().keys())[0] # getting the file from the local system
text = open(path_to_file, 'rb').read().decode(encoding='utf-8') # opening, reading and storing the file in variable text

vocab = sorted(set(text)) # getting all the unique characters present in our input file
# creating some constants that would be required later-
vocab_size = len(vocab)
batch_size = 128
embedding_dim = 256
rnn_units = 1024
buffer_size = 10000 # used to shuffle our batches in size of our buffer

# tokening texts into ints
char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

# Might as well create a decoder function at the same time
def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

# turning our int arrays into proper tensor datasets
text_as_int = text_to_int(text)
tensorDataset = tf.data.Dataset.from_tensor_slices(text_as_int)

# splitting the resulting data into training data and test data through this process-
""" n = int(0.9 * len(converted_data))
train_data = data[:n]
test_data = data[n:]
or use this approcach- """
# we are also first splitting our data into smaller sequences to make it easier for the model to process them-
seq_length = 8 # this is the size of the smaller dataset
examples_per_epoch = len(text) // (seq_length + 1)

sequences = tensorDataset.batch(seq_length+1, drop_remainder=True) # Now we have split our dataset into smaller pieces for easier processing

# Now its time to turn our data into input and output format. The way we split it into is as follows-
""" We split our data for e.g. Hello into as such- input = Hell; Expected output = Ello
This way we expect our model to predict the asked words + 1 """

def splitting_chunks(chunk):
  input = chunk[:-1]
  expected = chunk[1:]
  return input, expected

dataset = sequences.map(splitting_chunks) # This maps the splitting onto every batch of sequence

# Creating batches-
data = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

# Time for model creation
def model_creation(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
      tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
      tf.keras.layers.LSTM(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = model_creation(vocab_size, embedding_dim, rnn_units, batch_size)

# creating our own loss function as tensorflow doesn't have a built-in loss funtion which does the correct calculations for our varied (diff dim) losses 
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# compiling the model
model.compile(optimizer='adam', loss=loss)


