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

def train(x_train, y_train):
    dim = x_train.shape[1]
    cnt1, cnt2 = 0, 0
    mu1, mu2 = np.zeros((dim,)), np.zeros((dim,))

    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            cnt1 += 1
            mu1 += x_train[i]
        else:
            cnt2 += 1
            mu2 += x_train[i]
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((dim,dim))
    sigma2 = np.zeros((dim,dim))
    for i in range(x_train.shape[0]):
        if y_train[i] == 1:
            sigma1 += np.dot(np.transpose([x_train[i] - mu1]), [(x_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([x_train[i] - mu2]), [(x_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2

    share_sigma = (cnt1 / x_train.shape[0]) * sigma1 + (cnt2 / x_train.shape[0]) * sigma2
    return mu1, mu2, share_sigma, cnt1, cnt2

def predict(x_test, mu1, mu2, share_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(share_sigma)

    w = np.dot( (mu1-mu2), sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    pred = toclass(pred).astype(int)
    return pred

def save_prediction(pred, output_file):
    pred = toclass(pred)
    ans = pd.DataFrame({"id": np.arange(len(pred))+1, "label": pred})
    ans.to_csv(output_file,index=False)
    return

def main():
    if len(sys.argv) != 7:
        sys.exit("Usage: python3 $1 $2 $3 $4 $5 $6")
    # load data
    x_train = read_n_normalize(sys.argv[3])
    y_train = pd.read_csv(sys.argv[4], header=None).values
    x_test = read_n_normalize(sys.argv[5])

    # train
    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
    y = predict(x_train, mu1, mu2, shared_sigma, N1, N2)
    result = (y_train.reshape(-1) == y)
    print('Training accuracy = %f' % (float(result.sum()) / result.shape[0]))

    # predict
    pred = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
    save_prediction(pred, sys.argv[6])
    return

if __name__ == '__main__':
    main()
