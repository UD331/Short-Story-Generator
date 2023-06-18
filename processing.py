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

