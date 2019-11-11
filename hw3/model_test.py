import os
import cv2
import sys
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from datetime import datetime
from torchvision import transforms


def load_data(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()

    train_data = list(zip(train_image, train_label))
    random.shuffle(train_data)

    train_set = train_data[:28000]
    valid_set = train_data[28000:]

    return train_set, valid_set

def load_testdata(img_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    #train_label = pd.read_csv(label_path)
    #train_label = train_label.iloc[:,1].values.tolist()
    train_label = [i for i in range(len(train_image))]#

    train_data = list(zip(train_image, train_label))

    return train_data

def model_predict(model, data_dir):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    test_set = load_testdata(data_dir)
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = hw3_dataset(test_set,transform)
    test_loader = DataLoader(test_dataset,batch_size=128, shuffle=False)

    pred = np.array([])
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.data.cpu().numpy()
            pred = np.append(pred, predicted)
    return pred

class hw3_dataset(Dataset):

    def __init__(self, data, transform):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = Image.open(self.data[idx][0]).convert('RGB')
        img = self.transform(img)
        label = self.data[idx][1]
        return img, label

class Resnet34(nn.Module):
    def __init__(self):
        super(Resnet34, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet34(pretrained=True).children())[:-1])
        self.fc = nn.Linear(512,7)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1*1*512)
        x = self.fc(x)
        return x

class WideResnet50_2(nn.Module):
    def __init__(self):
        super(WideResnet50_2, self).__init__()
        self.resnet = nn.Sequential(*list(models.wide_resnet50_2(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048,7)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1*1*2048)
        x = self.fc(x)
        return x

class Resnet50(nn.Module):
    def __init__(self):
        super(Resnet50, self).__init__()
        self.resnet = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048,7)
    def forward(self, x):
        x = self.resnet(x)
        x = x.view(-1, 1*1*2048)
        x = self.fc(x)
        return x

def voting(pred1, pred2, pred3):
    '''
    Max Voting for 3 prediction result of classification problem.
    '''
    count = 0
    pred_final = pred1
    if len(pred1) == len(pred2) and len(pred2) == len(pred3):
        for i in range(len(pred1)):
            if pred1[i] != pred2[i] or pred2[i] != pred3[i]:
                if pred1[i] == pred2[i] and pred2[i] != pred3[i]:
                    pred_final[i] = pred1[i]
                elif pred1[i] == pred3[i] and pred2[i] != pred3[i]:
                    pred_final[i] = pred1[i]
                elif pred2[i] == pred3[i] and pred2[i] != pred1[i]:
                    pred_final[i] = pred2[i]
                else:
                    pred_final[i] = pred1[i]
    return pred_final

def save_pred(pred, file_name):
    idx = np.arange(len(pred))
    ans = pd.DataFrame(data={"id": idx, "label": pred}, dtype="int64")
    ans.to_csv(sys.argv[2], index=False)
    return

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 $1 $2")

    model1 = Resnet34()
    model1.load_state_dict(torch.load('20191106_res34_model_57.pth'))
    pred1 = model_predict(model1, sys.argv[1])
    print("Loaded model 1")

    model2 = WideResnet50_2()
    model2.load_state_dict(torch.load('20191105_wideres50_model_146.pth'))
    pred2 = model_predict(model2, sys.argv[1])
    print("Loaded model 2")

    model3 = Resnet50()
    model3.load_state_dict(torch.load('20191106_res50_model_112.pth'))
    pred3 = model_predict(model3, sys.argv[1])
    print("Loaded model 3")

    pred = voting(pred1, pred2, pred3)
    save_pred(pred, sys.argv[2])
    return




if __name__ == '__main__':
    main()
