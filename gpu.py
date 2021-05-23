# Import library
#######################################################
import tensorflow as tf
from tensorflow import keras
import datetime
import os

import logging

# logging parameter
logging.basicConfig(level=logging.ERROR)

#######################################################
# Set GPU
#######################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

#######################################################
# Set Memory
#######################################################
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)