import torch
import torchvision.transforms as transforms
import torch.utils.data as data

from os.path import join, exists
from scipy.io import loadmat
import numpy as np
from random import randint, random
from collections import namedtuple
from PIL import Image
from os import listdir
from os.path import isfile, join
from sklearn.neighbors import NearestNeighbors
import h5py

def input_transform():
    return transforms.Compose([transforms.Resize([240,240], interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        
    ])

def get_whole_training_set(onlyDB=False):
    return WholeDatasetFromStruct(input_transform=input_transform(),
                             onlyDB=onlyDB)

class WholeDatasetFromStruct(data.Dataset):
    def __init__(self, input_transform=None, onlyDB=False):
        super().__init__()
        #print('zzzzzzz!!~~~')
        self.input_transform = input_transform
        self.images = ['/home/tourani/Desktop/output/captures/'+f for f in listdir('/home/tourani/Desktop/output/captures/') if isfile(join('/home/tourani/Desktop/output/captures/', f))]
        #self.images = 
        self.images.sort()
        #print(self.images)
        #print(self.images)corridor_new_every_4th
        self.whichSet = 'test'
        self.dataset = None

        self.positives = None
        self.distances = None
        #print(onlyfiles)
    def __getitem__(self, index):
        img = Image.open(self.images[index])
        if self.input_transform:
            img = self.input_transform(img)
        #print(img.shape, 'kuyyan')
        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        #fit NN to find them, search by radius
        if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                    radius=self.dbStruct.posDistThr)

        return self.positives
