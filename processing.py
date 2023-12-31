import tensorflow as tf
import numpy as np
import os
import keras
from keras.preprocessing import sequence
from google.colab import files

# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

path_to_file = list(files.upload().keys())[0] # getting the file from the local system
# Using The Picture of Dorian Gray in this case

temp = open(path_to_file, 'rb')
pdfreader=PyPDF2.PdfReader(temp)

#This will store the number of pages of this pdf file
x=len(pdfreader.pages)

#create a variable that will select the selected number of pages
for i in range(0, x):
  pageobj=pdfreader.pages[i]

#(x+1) because python indentation starts with 0.
#create text variable which will store all text datafrom pdf file
  text=pageobj.extract_text()

#save the extracted data from pdf to a txt file
#we will use file handling here
#dont forget to put r before you put the file path
#go to the file location copy the path by right clicking on the file
#click properties and copy the location path and paste it here.
#put "\\your_txtfilename"

  file1=open(r"1.txt","a")
  file1.writelines(text)

file1.close()
f = open("1.txt", "rb")
text = f.read().decode(encoding='utf-8')

# text = open(path_to_file, 'rb').read().decode(encoding='utf-8') # opening, reading and storing the file in variable text

vocab = sorted(set(text)) # getting all the unique characters present in our input file
# creating some constants that would be required later-
vocab_size = len(vocab)
batch_size = 64 # 128 too big, start with smaller batch sizes of 32/64
# smaller batches also want smaller learning rates
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
seq_length = 100 # this is the size of the smaller dataset. Take it small if we want to analyze sentences like in this case or larger if we
# want to analyze documents
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
      tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
      tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = model_creation(vocab_size, embedding_dim, rnn_units, batch_size)

# creating our own loss function as tensorflow doesn't have a built-in loss funtion which does the correct calculations for our varied (diff dim) losses 
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

# compiling the model
model.compile(optimizer='adam', loss=loss)

# setting up checkpoints to save our model training data to and allowing us to use it at any point in the future as callback to train our model
checkpoint_dir = './checkpoints'
# name of checkpoint files for each epoch
check_name = os.path.join(checkpoint_dir, "c_{epoch}")
# now we are trying to setup our way of saving checkpoints
check_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_name, save_weights_only=True)

# Training the model now
history = model.fit(data, epochs= 10, callbacks=[check_callback])

# After we train our model, we retrain it with a batch size of 1 and run it again from various checkpoints (usually just calling the last checkpoint)
model = model_creation(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Now we will be generating text using generate text function
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 8000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

# Now time to test it
inp = input("Type a starting string: ")
print(generate_text(model, inp))
