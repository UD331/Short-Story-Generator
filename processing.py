import tensorflow as tf
import numpy as np
import os
import keras
from keras.preprocessing import sequence
from google.colab import files

path_to_file = list(files.upload().keys())[0]
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
