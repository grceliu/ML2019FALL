import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from sklearn.model_selection import KFold
from keras import regularizers

def readdata(file_path):
    data = pd.read_csv(file_path)
    data["net_gain"] = (data["capital_gain"] - data["capital_loss"] > 0).astype("int")
    #data["is_adult"] = (data["age"] >= 21).astype("int")
    #data = data[[col for col in data if not col.startswith('?_')]]
    data = data.values
    return data

def normalize(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)

    index = [0, 1, 3, 4, 5]
    mean_vec = np.zeros(x_all.shape[1])
    std_vec = np.ones(x_all.shape[1])
    mean_vec[index] = mean[index]
    std_vec[index] = std[index]

    x_all_nor = (x_all - mean_vec) / std_vec

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor

def train(X_TRAIN, Y_TRAIN):
    model = Sequential()
    model.add(Dense(30, input_dim=x_train.shape[1], kernel_regularizer=regularizers.l2(0.001), init='uniform', activation='relu'))
    for i in range(6):
        model.add(Dense(100, kernel_regularizer=regularizers.l2(0.001), init='uniform', activation='relu'))
        Dropout(rate=0.5, seed=0)
    model.add(Dense(1, init='uniform', activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_TRAIN, Y_TRAIN, nb_epoch=100, batch_size=100, verbose=1)
    model.save("bashtest_PassStrong.h5")
    return

def toclass(X):
    return np.around(X)

def predict_n_save(model, data, output_file):
    pred = toclass(model.predict(data)).astype(int).flatten()
    ans = pd.DataFrame({"id": np.arange(len(pred))+1, "label": pred})
    ans.to_csv(output_file,index=False)
    return

def main():
    if len(sys.argv) != 7:
        sys.exit("Usage: python3 $1 $2 $3 $4 $5 $6")
    # load data
    x_train = readdata(sys.argv[3])
    y_train = pd.read_csv(sys.argv[4], header=None).values
    x_test = readdata(sys.argv[5])
    x_train, x_test = normalize(x_train, x_test)

    # X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
    # train(X_TRAIN, Y_TRAIN)

    model = load_model("FirstPassStrong.h5")
    predict_n_save(model, x_test, sys.argv[6])
    return

if __name__ == '__main__':
    main()
