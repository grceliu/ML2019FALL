import re
import os
import sys
import spacy
import pickle
import en_core_web_sm

import numpy as np
import pandas as pd

from datetime import datetime
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
TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

def main():
    if len(sys.argv) != 4:
        sys.exit("Usage: python3 train_hw4.py $1 $2 $3")

    # read data
    x_train = pd.read_csv(sys.argv[1]).values
    y_train = pd.read_csv(sys.argv[2]).values
    x_test = pd.read_csv(sys.argv[3]).values

    nlp = en_core_web_sm.load()
    sentences = x_train[:,1]
    for i in range(len(sentences)):
        doc = nlp(sentences[i])
        sentences[i] = [t.text for t in doc]
    max_length = max([len(sentence) for sentence in sentences])
    vocab_size = len(sentences)
    EMBEDDING_DIM = 400

    # Train & Save Word2Vec model
    model = Word2Vec(min_count=1, size = EMBEDDING_DIM, workers=4)
    model.build_vocab(sentences)  # prepare the model vocabulary
    model.train(sentences, total_examples=model.corpus_count, epochs=20)  # train word vectors

    # save model
    filename = 'word2vec_{}.txt'.format(TIME)
    model.wv.save_word2vec_format(filename, binary=False)

    # read the model
    embeddings_index = {}
    f = open(os.path.join('', filename), encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs
    f.close

    x_train = list(x_train[:,1])

    # Train & Save Tokenizer
    tokenizer_obj = Tokenizer()
    tokenizer_obj.fit_on_texts(x_train)

    # save model
    pickle_name = 'tokenizer_{}.txt'.format(TIME)
    with open(pickle_name, 'wb') as handle:
        pickle.dump(tokenizer_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # loading
    with open(pickle_name, 'rb') as handle:
        tokenizer_obj = pickle.load(handle)

    sequences = tokenizer_obj.texts_to_sequences(x_train)
    word_index = tokenizer_obj.word_index
    review_pad = pad_sequences(sequences, maxlen=max_length)
    target = y_train[:,1]

    num_words = len(word_index) + 1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # split to training and testing data
    VALIDATION_SPLIT = 0.01

    indices = np.arange(review_pad.shape[0])
    np.random.shuffle(indices)
    review_pad = review_pad[indices]
    target = target[indices]
    num_validation_samples = int(VALIDATION_SPLIT * review_pad.shape[0])

    x_train_pad = review_pad[:-num_validation_samples]
    y_train = target[:-num_validation_samples]

    x_test_pad = review_pad[-num_validation_samples:]
    y_test = target[-num_validation_samples:]

    # model1 - GRU
    model= Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)

    model.add(embedding_layer)
    model.add(GRU(units=50, dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
    model.add(GRU(units=50, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pad, y_train, batch_size=128, epochs=1, validation_data=(x_test_pad, y_test), verbose=2, shuffle=True)
    model.save('model_{}.h5'.format(TIME))

    # model2: Bi-GRU
    model= Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)

    model.add(embedding_layer)
    model.add(Bidirectional(GRU(units=40, dropout=0.4, recurrent_dropout=0.4,return_sequences=True)))
    model.add(Bidirectional(GRU(units=40, dropout=0.4, recurrent_dropout=0.4)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pad, y_train, batch_size=128, epochs=1, validation_data=(x_test_pad, y_test), verbose=2, shuffle=True)
    model.save('model_{}.h5'.format(TIME))

    # model3: Bi-GRU
    model= Sequential()
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=max_length,
                                trainable=False)

    model.add(embedding_layer)
    model.add(Bidirectional(GRU(units=30, dropout=0.5, recurrent_dropout=0.5,return_sequences=True)))
    model.add(Bidirectional(GRU(units=30, dropout=0.5, recurrent_dropout=0.5)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train_pad, y_train, batch_size=128, epochs=1, validation_data=(x_test_pad, y_test), verbose=2, shuffle=True)
    model.save('model_{}.h5'.format(TIME))

    print("TRAINING COMPLETED")

if __name__ == '__main__':
    main()
