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

def load_testdata(img_path, label_path):
    train_image = sorted(glob.glob(os.path.join(img_path, '*.jpg')))
    train_label = pd.read_csv(label_path)
    train_label = train_label.iloc[:,1].values.tolist()

    train_data = list(zip(train_image, train_label))

    return train_data

def model_predict(model, data_dir):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    test_set = load_testdata(data_dir, 'train.csv') # train.csv is useless here XD
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

def train_model(model, num_epoch, model_name, train_loader, valid_loader):
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epoch):
        model.train()
        train_loss = []
        train_acc = []
        for idx, (img, label) in enumerate(train_loader):
            if use_gpu:
                img = img.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            output = model(img)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            predict = torch.max(output, 1)[1]
            acc = np.mean((label == predict).cpu().numpy())
            train_acc.append(acc)
            train_loss.append(loss.item())
        print("Epoch: {}, train Loss: {:.4f}, train Acc: {:.4f}".format(epoch + 1, np.mean(train_loss), np.mean(train_acc)))


        model.eval()
        with torch.no_grad():
            valid_loss = []
            valid_acc = []
            for idx, (img, label) in enumerate(valid_loader):
                if use_gpu:
                    img = img.cuda()
                    label = label.cuda()
                output = model(img)
                loss = loss_fn(output, label)
                predict = torch.max(output, 1)[1]
                acc = np.mean((label == predict).cpu().numpy())
                valid_loss.append(loss.item())
                valid_acc.append(acc)
            print("Epoch: {}, valid Loss: {:.4f}, valid Acc: {:.4f}".format(epoch + 1, np.mean(valid_loss), np.mean(valid_acc)))
    torch.save(model.state_dict(), model_name)
    print('model saved to %s' % model_name)
    return

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

def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 $1 $2")

    #transform to tensor, data augmentation
    transform = transforms.Compose([transforms.ToTensor()])

    train_set, valid_set = load_data(sys.argv[1], sys.argv[2])
    #load data
    train_dataset = hw3_dataset(train_set,transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    valid_dataset = hw3_dataset(valid_set,transform)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)

    # train model
    print("training model 1...")
    model1 = Resnet34()
    train_model(model1, 57, "20191106_res34_model_57.pth", train_loader, valid_loader)
    print("training model 2...")
    model2 = WideResnet50_2()
    train_model(model2, 146, "20191105_wideres50_model_146.pth", train_loader, valid_loader)
    print("training model 3...")
    model3 = Resnet50()
    train_model(model3, 112, "20191106_res50_model_112.pth", train_loader, valid_loader)

    return


if __name__ == '__main__':
    main()
