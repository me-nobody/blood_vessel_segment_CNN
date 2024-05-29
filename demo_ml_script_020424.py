# demo ML script for SLURM



import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import logging


log = logging.getLogger("my-logger")
log.info("Hello, world")

