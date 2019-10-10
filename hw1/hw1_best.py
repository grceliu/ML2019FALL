import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def readdata(data):
    # 把有些數字後面的奇怪符號刪除
    for col in list(data.columns[2:]):
        data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
    data = data.values

    # 刪除欄位名稱及日期
    data = np.delete(data, [0,1], 1)

    # 特殊值補0
    data[ data == 'NR'] = 0
    data[ data == ''] = 0
    data[ data == 'nan'] = 0
    data = data.astype(np.float)

    return data

def extract(data):
    N = data.shape[0] // 18

    temp = data[:18, :]

    # Shape 會變成 (x, 18) x = 取多少hours
    for i in range(1, N):
        temp = np.hstack((temp, data[i*18: i*18+18, :]))
    return temp

def valid(x, y):
    if y <= 2 or y > 100:
        return False
    for i in range(9):
        if x[9,i] <= 2 or x[9,i] > 100:
            return False
    return True

def parse2train(data):
    x = []
    y = []

    # 用前面9筆資料預測下一筆PM2.5 所以需要-9
    total_length = data.shape[1] - 9
    for i in range(total_length):
        x_tmp = data[:,i:i+9]
        y_tmp = data[9,i+9]
        if valid(x_tmp, y_tmp):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)
    # x 會是一個(n, 18, 9)的陣列， y 則是(n, 1)
    x = np.array(x)
    y = np.array(y)
    return x,y

def normalize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    for i in range(len(X)):
        for j in range(len(X[0])):
            X[i][j] = (X[i][j] - mean[j]) / std[j]
    return X, mean, std

def gradient_descent(X, Y, x_test, y_test, update_times):
    # gradient descent
    dim = X.shape[1]

    w = np.zeros(shape = (dim, 1))

    lr_rate = np.array([[0.1]] * dim)
    adagrad_sum = np.zeros(shape = (dim, 1))

    for i in range(update_times):
        gradient = (-2) * X.T.dot(Y - X.dot(w))
        adagrad_sum += np.square(gradient)
        w = w - lr_rate * gradient / (np.sqrt(adagrad_sum)+0.0000005)

        if i % 10000 == 0:
            print("Update", i, " RMSE: training: ", rmse(Y,X.dot(w)), " testing: ", rmse(y_test, x_test.dot(w)))
    return w

def rmse(y, y_pred):
    root_mean_squared_error = np.sqrt(np.square(y - y_pred).sum() / len(y_pred))
    return root_mean_squared_error

def train():
    # training
    yr1 = pd.read_csv("./data/year1-data.csv")
    yr2 = pd.read_csv("./data/year2-data.csv")
    data = pd.concat([yr1, yr2], axis=0)
    data = readdata(data)
    train_data = extract(data)
    X, Y = parse2train(train_data)
    X, train_mean, train_std = normalize(X)
    Y = np.reshape(Y, (-1, 1))
    X = np.concatenate([X, np.ones(shape=(X.shape[0],1))], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.5, random_state=0, shuffle=True)
    w = gradient_descent(x_train, y_train, x_test, y_test, update_times=200000)

    print("final: ", rmse(y=Y, y_pred=X.dot(w)))
    np.save("best_weight.npy", w)
    np.save("mean.npy", train_mean)
    np.save("std.npy", train_std)

#############################
############start############
#############################
if len(sys.argv) != 3:
    sys.exit("Usage: python3 hw1.py [input_file] [output_file]")

# train()

# testing
df = pd.read_csv(sys.argv[1])
df = df.replace("NR", 0)
for i in range(9):
    df[str(i)] = df[str(i)].replace(to_replace ="[^0-9.]", value = "", regex = True)
    df[str(i)] = df[str(i)].astype("float64")
df = df.fillna(0)

sta_id = df["id"].unique()

X = df.iloc[:, 2:].values
X = np.reshape(X, (len(sta_id), -1))

train_mean = np.load("mean.npy")
train_std = np.load("std.npy")

for i in range(len(X)):
    for j in range(len(X[0])):
        X[i][j] = (X[i][j] - train_mean[j]) / train_std[j]

X = np.concatenate([X, np.ones(shape=(X.shape[0],1))], axis=1)

w = np.load('best_weight.npy')

pred = X.dot(w)
pred = pred.flatten()
ans = pd.DataFrame({"id": sta_id, "value": pred})
ans.value = ans.value.apply(lambda x: max(x, 0))
ans.to_csv(sys.argv[2],index=False)
