import os
import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from torch.cuda.amp import autocast, GradScaler

class Dataset2class(torch.utils.data.Dataset):
    def __init__(self, path_dir1: str, path_dir2: str):
        super().__init__()

        self.path_dir1 = path_dir1
        self.path_dir2 = path_dir2

        self.dir1_list = sorted(os.listdir(path_dir1))
        self.dir2_list = sorted(os.listdir(path_dir2))

    def getImageIndex(self, index):
        if index < len(self.dir1_list):
            path = os.path.join(self.path_dir1, self.dir1_list[index])
            id = 0
        else:
            path = os.path.join(self.path_dir2, self.dir2_list[index - len(self.dir1_list)])
            id = 1
        return path, id

    def __getitem__(self, index):
        img = None
        sub_index = 0
        while img is None:
            path, id_class = self.getImageIndex(index + sub_index)
            sub_index += 1
            img = cv.imread(path, cv.IMREAD_COLOR)
            if img is not None:
                break
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = img / 255.0
        img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
        img = img.transpose((2, 0, 1))

        t_img = torch.from_numpy(img)
        t_id = torch.tensor(id_class)

        return {'img': t_img, 'label': t_id}

    def __len__(self):
        return len(self.dir1_list) + len(self.dir2_list)

train_dogs_path = 'imageFolder/trainingDataSetDogs/'
train_cats_path = 'imageFolder/trainingDataSetCats/'
test_dogs_path = 'imageFolder/Dog/'
test_cats_path = 'imageFolder/Cat/'

train_ds_catsdogs = Dataset2class(train_dogs_path, train_cats_path)
test_ds_catsdogs = Dataset2class(test_dogs_path, test_cats_path)

batch_size = 16

train_loader = torch.utils.data.DataLoader(
    train_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_ds_catsdogs, shuffle=True,
    batch_size=batch_size, num_workers=0, drop_last=False
)

model = tv.models.resnet34(num_classes=2)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)
loss_fn = loss_fn.to(device)

use_amp = True
scaler = torch.cuda.amp.GradScaler()

epochs = 10
loss_epochs_list = []
acc_epochs_list = []
for epoch in range(epochs):
    loss_val = 0
    acc_val = 0
    for sample in (pbar := tqdm(train_loader)):
        img, label = sample['img'], sample['label']
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        with autocast(use_amp):
            pred = model(img)
            loss = loss_fn(pred, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_item = loss.item()
        loss_val += loss_item

        acc_current = (pred.argmax(1) == label).float().mean().item()
        acc_val += acc_current

        pbar.set_description(f'loss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')
    loss_epochs_list.append(loss_val / len(train_loader))
    acc_epochs_list.append(acc_val / len(train_loader))
    print(f'Epoch {epoch + 1}:')
    print(f'  Loss: {loss_epochs_list[-1]}')
    print(f'  Accuracy: {acc_epochs_list[-1]}')

# После обучения модели можно провести тестирование

loss_val = 0
acc_val = 0
for sample in (pbar := tqdm(test_loader)):
    with torch.no_grad():
        img, label = sample['img'], sample['label']
        img = img.to(device)
        label = label.to(device)

        pred = model(img)
        loss = loss_fn(pred, label)
        loss_val += loss.item()

        acc_current = (pred.argmax(1) == label).float().mean().item()
        acc_val += acc_current

    pbar.set_description(f'loss: {loss.item():.5f}\taccuracy: {acc_current:.3f}')

print(f'Test Results:')
print(f'  Loss: {loss_val / len(test_loader)}')
print(f'  Accuracy: {acc_val / len(test_loader)}')
