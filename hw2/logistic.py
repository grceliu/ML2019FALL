import sys
import numpy as np
import pandas as pd

def read_n_normalize(file_path):
    data = pd.read_csv(file_path)
    continuous = []
    for col in data:
        if len(data[col].unique()) > 2:
            continuous.append(col)
    df = data[continuous]
    df = (df - df.mean()) / df.std()
    data[continuous] = df
    data = data.values
    return data

def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def accuracy(y_pred, y_true):
    acc = (y_pred == y_true).sum() / len(y_pred)
    return acc

def toclass(X):
    return np.around(X)

def train_logistic(X, Y, epoch=20000, batch=2000, lr=0.1, test_size=0.3):
    # random sampling
    rand = np.random.choice(len(X), int((1-test_size) * len(X)))
    X_test, Y_test = X[~rand], Y[~rand]
    X, Y = X[rand], Y[rand]

    # gradient descent
    dim = X.shape[1]
    w = np.zeros(shape = (dim, 1))
    lr_rate = np.array([[lr]] * dim)
    adagrad_sum = np.zeros(shape = (dim, 1))

    for i in range(epoch):
        x = X[epoch*batch%len(X) : (epoch*batch+batch)%len(X)]
        y = Y[epoch*batch%len(Y) : (epoch*batch+batch)%len(Y)]

        gradient = (-1) * x.T.dot(y - sigmoid(x.dot(w)))
        adagrad_sum += np.square(gradient)
        w = w - lr_rate * gradient / (np.sqrt(adagrad_sum)+0.00000000000001)

        if (i+1) % 100== 0:
            print("Update", i+1, " train acc: ", accuracy(Y,toclass(sigmoid(X.dot(w)))), "test acc: ", accuracy(Y_test,toclass(sigmoid(X_test.dot(w)))))
    return w

def train(x_train, y_train):
    w = train_logistic(x_train, y_train, epoch=2000, batch=10000)
    np.save("SimpleBaseline_weight.npy", w)
    return

def predict(x_test, output_file):
    w = np.load("SimpleBaseline_weight.npy")
    pred = toclass(sigmoid(x_test.dot(w)))
    pred = pred.flatten().astype(int)
    ans = pd.DataFrame({"id": np.arange(len(pred))+1, "label": pred})
    ans.to_csv(output_file,index=False)
    return

def main():
    if len(sys.argv) != 7:
        sys.exit("Usage: python3 $1 $2 $3 $4 $5 $6")
    x_train = read_n_normalize(sys.argv[3])
    y_train = pd.read_csv(sys.argv[4], header=None).values
    x_test = read_n_normalize(sys.argv[5])
    #train(x_train, y_train)
    y_pred = predict(x_test, sys.argv[6])

if __name__ == '__main__':
    main()
