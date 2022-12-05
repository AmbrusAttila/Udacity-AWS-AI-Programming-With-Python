import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import json
import pandas as pd
import numpy as np

from PIL import Image

def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    BATCH_SIZE=64

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    valid_test_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transforms)


    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    return trainloader, validloader, testloader

def load_categories(category_path):
    with open(category_path, 'r') as f:
        cat_to_name = json.load(f)

    df_cls=pd.DataFrame({"class": cat_to_name})

    return df_cls


def process_image(image_path):        
    with Image.open(image_path) as im:
        im=im.resize((256,256)) 
        im=im.crop((16, 16, 240, 240))
        np_image = (np.array(im)/255)[:,:,0:3]
        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])        
        np_image=(np_image-mean)/std
        np_image=np_image.transpose((2, 0, 1))        
        
        return np_image
