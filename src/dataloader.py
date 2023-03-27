from torch.utils.data import Dataset, DataLoader
from config import cfg
import os
import random
import matplotlib.pyplot as plt
from matplotlib import image
import torch
import numpy as np
import torchvision
from PIL import Image


class image_dataset(Dataset):

    def __init__(self, data_paths, label_paths, train):
        self.data_paths = data_paths
        self.label_paths = label_paths

        self.train_mode = train

    def __getitem__(self, idx):
        x_data = torch.tensor(image.imread(self.data_paths[idx]))
        x_data = torch.swapaxes(x_data, -1, 0)
        x_data = torch.swapaxes(x_data, -1, 1)

        y_data = torch.tensor(image.imread(self.label_paths[idx]))
        y_data = torch.mean(y_data, dim=-1)
        y_data = y_data.type(torch.int64)

        y_data = torch.nn.functional.one_hot(y_data, cfg.DATASET.NBR_CLASSE).float()
        y_data = torch.swapaxes(y_data, -1, 0)
        y_data = torch.swapaxes(y_data, -1, 1)

        if self.train_mode:
            x_data, y_data = self.augmentation(x_data, y_data)

        return x_data.float(), y_data
    
    def augmentation(self, data, label):
        
        if np.random.rand() > 0.5:
            data = torch.flip(data, [1])
            label = torch.flip(label, [1])
        
        if np.random.rand() > 0.5:
            data = torch.flip(data, [2])
            label = torch.flip(label, [2])

        nbr_rotation = np.random.randint(0,2)*2
        data = torch.rot90(data, nbr_rotation, [1,2])
        label = torch.rot90(label, nbr_rotation, [1,2])

        return data, label
    
    def __len__(self):
        return len(self.data_paths)
    

def get_dataloader():

    data_paths = []
    label_paths = []
    for image_path in os.listdir(cfg.DATASET.TRAIN_DATA_FOLDER_PATH):

        if image_path[-3:] == 'png':
            continue

        if os.path.isfile(os.path.join(cfg.DATASET.TRAIN_DATA_FOLDER_PATH, image_path.split('_')[0] + '_mask.png')):
            label_paths.append(os.path.join(cfg.DATASET.TRAIN_DATA_FOLDER_PATH, image_path.split('_')[0] + '_mask.png'))
            data_paths.append(os.path.join(cfg.DATASET.TRAIN_DATA_FOLDER_PATH, image_path))


    train_dataloader = DataLoader(image_dataset(data_paths[:int(cfg.DATASET.TRAIN_PROPORTION*len(data_paths))], label_paths[:int(cfg.DATASET.TRAIN_PROPORTION*len(data_paths))], True), batch_size=cfg.DATASET.TRAINING_BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(image_dataset(data_paths[int(cfg.DATASET.TRAIN_PROPORTION*len(data_paths)):int((cfg.DATASET.TRAIN_PROPORTION+cfg.DATASET.VALIDATION_PROPORTION)*len(data_paths))], label_paths[int(cfg.DATASET.TRAIN_PROPORTION*len(data_paths)):int((cfg.DATASET.TRAIN_PROPORTION+cfg.DATASET.VALIDATION_PROPORTION)*len(data_paths))], False), batch_size=cfg.DATASET.VALIDATION_BATCH_SIZE, shuffle=False)
    test_dataloader = DataLoader(image_dataset(data_paths[int((cfg.DATASET.TRAIN_PROPORTION+cfg.DATASET.VALIDATION_PROPORTION)*len(data_paths)):], label_paths[int((cfg.DATASET.TRAIN_PROPORTION+cfg.DATASET.VALIDATION_PROPORTION)*len(data_paths)):], False), batch_size=cfg.DATASET.TEST_BATCH_SIZE, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader


def show_batch():
    train_dataloader, val_dataloader, test_dataloader = get_dataloader()

    for i, (datas, labels) in enumerate(train_dataloader):
        for i in range(len(datas)):
            img = torch.swapaxes(datas[i], -1, 0)
            img = torch.swapaxes(img, 1, 0).type(torch.uint8)

            img = img.numpy()
            img = Image.fromarray(np.uint8(img))

            img.save(f'./tmp/img_{i}.png')
        
        for i in range(len(labels)):

            img = torch.tensor(labels[i])
            img = torch.argmax(img, dim=-1)
            img = torch.unsqueeze(img, dim=-1)
            img = torch.cat([img,img,img], dim=-1)

            img = img*255
            img = img.numpy()
            img = Image.fromarray(np.uint8(img))

            img.save(f'./tmp/img_label_{i}.png')
        break

