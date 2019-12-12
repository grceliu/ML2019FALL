import re
import os
import sys
import spacy
import pickle
import en_core_web_sm

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from keras import regularizers
from keras.models import load_model
from keras.models import Sequential, Model
from keras.layers import *
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.regularizers import l2
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

def model_predict(model_path, pickle_path, data_path):
    max_length = 112
    model = load_model(model_path)
    x_test = pd.read_csv(data_path).values
    x_test = list(x_test[:,1])

    with open(pickle_path, 'rb') as handle:
        tokenizer_obj = pickle.load(handle)
    test_samples_tokens = tokenizer_obj.texts_to_sequences(x_test)
    test_samples_tokens_pad = pad_sequences(test_samples_tokens, maxlen=max_length)

    pred = model.predict(x=test_samples_tokens_pad)
    pred = np.around(pred.flatten())

    return pred

def voting(*args):
    '''
    max voting for binary classification, uniform weight
    input: 1-n arrays of predictions to vote
    output: 1 array of final result
    '''
    num = len(args[0])
    pred_final = np.zeros(num)
    for i in range(num):
        vote = 0
        for arg in args:
            vote += arg[i]
        if vote >= len(args)/2:
            pred_final[i] = 1
        else:
            pred_final[i] = 0
    return pred_final

def save_pred(pred, file_name):
    df = pd.DataFrame({'id': np.arange(0,len(pred)), 'label': pred}, dtype="int64")
    df.to_csv(file_name,index=False)
    return

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 test_hw4.py $1 $2")

    pred1 = model_predict("model_20191209_1.h5","tokenizer_1209.pickle", sys.argv[1])
    pred2 = model_predict("model_20191209_2.h5","tokenizer_1209.pickle", sys.argv[1])
    pred3 = model_predict("model_20191211.h5", "tokenizer_1211.pickle",sys.argv[1])

    pred = voting(pred1, pred2, pred3)
    save_pred(pred, sys.argv[2])
    return

if __name__ == '__main__':
    main()
