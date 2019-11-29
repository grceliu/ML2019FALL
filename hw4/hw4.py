import sys
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import optim
from sklearn.decomposition import DictionaryLearning
from torch.utils.data import DataLoader, Dataset
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # define: encoder
        self.encoder = nn.Sequential(
          nn.Conv2d(3, 64, 3,1,1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(64),

          nn.Conv2d(64, 128,  4,2,1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(128),

          nn.Conv2d(128, 256, 4,2,1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(256),
        )

        # define: decoder
        self.decoder = nn.Sequential(

          nn.ConvTranspose2d(256, 128, 4,2,1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(128),

          nn.ConvTranspose2d(128, 16,4,2,1),
          nn.LeakyReLU(),
          nn.BatchNorm2d(16),

          nn.ConvTranspose2d(16, 3, 3, 1, 1),
          nn.LeakyReLU(),
        )


    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        # Total AE: return latent & reconstruct
        return encoded, decoded

def model1(test_dataloader, autoencoder):
    latents = []
    reconstructs = []
    for x in test_dataloader:

        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())

    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

    # Use PCA to lower dim of latents and use K-means to clustering.
    latents = PCA(n_components=200, random_state=5).fit_transform(latents)
    latents = FactorAnalysis(n_components=50, random_state=30).fit_transform(latents)
    result = TSNE(n_components=2, random_state=9).fit_transform(latents)
    result = MiniBatchKMeans(n_clusters = 2, random_state=6).fit(result).labels_

    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    return result

def model2(test_dataloader, autoencoder):
    latents = []
    reconstructs = []
    for x in test_dataloader:

        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())

    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

    # Use PCA to lower dim of latents and use K-means to clustering.
    latents = PCA(n_components=160, random_state=5).fit_transform(latents)
    latents = FactorAnalysis(n_components=80, random_state=30).fit_transform(latents)
    result = TSNE(n_components=2, random_state=9).fit_transform(latents)
    result = MiniBatchKMeans(n_clusters = 2, random_state=6).fit(result).labels_

    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    return result

def model3(test_dataloader, autoencoder):
    latents = []
    reconstructs = []
    for x in test_dataloader:

        latent, reconstruct = autoencoder(x)
        latents.append(latent.cpu().detach().numpy())
        reconstructs.append(reconstruct.cpu().detach().numpy())

    latents = np.concatenate(latents, axis=0).reshape([9000, -1])
    latents = (latents - np.mean(latents, axis=0)) / np.std(latents, axis=0)

    # Use PCA to lower dim of latents and use K-means to clustering.
    latents = PCA(n_components=100, random_state=5).fit_transform(latents)
    latents = FactorAnalysis(n_components=50, random_state=30).fit_transform(latents)
    result = TSNE(n_components=2, random_state=9).fit_transform(latents)
    result = AgglomerativeClustering(n_clusters = 2).fit(result).labels_

    # We know first 5 labels are zeros, it's a mechanism to check are your answers
    # need to be flipped or not.
    if np.sum(result[:5]) >= 3:
        result = 1 - result
    return result


def voting(*args):
    '''
    max voting for binary classification, uniform weight
    input: 1-n array(s) of predictions to vote
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



def main():
    if len(sys.argv) != 3:
        sys.exit("Usage: python3 $1 $2")

    # detect is gpu available.
    use_gpu = torch.cuda.is_available()

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load('20191128_epoch200.pth'))

    # load data and normalize to [-1, 1]
    trainX = np.load(sys.argv[1])
    trainX = np.transpose(trainX, (0, 3, 1, 2)) / 255. * 2 - 1
    trainX = torch.Tensor(trainX)

    # if use_gpu, send model / data to GPU.
    if use_gpu:
        autoencoder.cuda()
        trainX = trainX.cuda()

    # Dataloader
    test_dataloader = DataLoader(trainX, batch_size=32, shuffle=False)

    # predict and ensemble
    pred1 = model1(test_dataloader, autoencoder)
    pred2 = model2(test_dataloader, autoencoder)
    pred3 = model3(test_dataloader, autoencoder)
    result = voting(pred1, pred2, pred3)

    # save result
    df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result}, dtype="int64")
    df.to_csv(sys.argv[2],index=False)

    return

if __name__ == '__main__':
    main()
