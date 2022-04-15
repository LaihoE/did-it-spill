import time
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import os
import torchvision
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.io import read_image

# NOT DONE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms_x = torch.nn.Sequential(
    T.Resize((64, 64)),
    T.ConvertImageDtype(torch.float),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
)
scripted_transforms = torch.jit.script(transforms_x)


class imagedataset(Dataset):
    def __init__(self):
        folder = '/mnt/f/amp/a/'
        files = os.listdir(folder)
        self.files = [f"{folder}{f}" for f in files]
        self.n_samples = len(files) - 1
        self.scripted_transforms = scripted_transforms

    def __getitem__(self, index):
        og_image = read_image(self.files[index])
        normal_image = self.scripted_transforms(og_image)
        return normal_image

    def __len__(self):
        return self.n_samples


class testdataset(Dataset):
    def __init__(self):
        folder = '/mnt/f/test/a/'
        files = os.listdir(folder)
        self.files = [f"{folder}{f}" for f in files]
        self.n_samples = len(files) - 1
        self.scripted_transforms = scripted_transforms

    def __getitem__(self, index):
        og_image = read_image(self.files[index])
        normal_image = self.scripted_transforms(og_image)
        return normal_image

    def __len__(self):
        return self.n_samples


def generate_embeddings(loader, model):
    # returns embeddings in shape (n, 1000)
    embeddings = []
    for data in loader:
        print(data.shape)
        pred2 = model(data.to(device))
        embeddings.append(pred2.flatten(start_dim=1).detach().cpu())
    return torch.cat(embeddings)


def knn(emb_train, emb_test, K, batch_size, threshold=30, print_dupes=False):
    for binx in range(math.ceil(len(emb_test) / batch_size)):
        batch_test_emb = emb_test[binx * batch_size: binx * batch_size + batch_size, :]
        D = torch.cdist(batch_test_emb, emb_train)
        dist, idx = D.topk(k=K, dim=-1, largest=False)

        if print_dupes:
            for i, row in enumerate(idx):
                for j, elem in enumerate(row):
                    if float(dist[i][j]) < threshold and files[binx * batch_size + i] != files[idx[i][j]]:
                        print(f"Image {files[binx * batch_size + i]}: -> {files[idx[i][j]]}", float(dist[i][j]))
        else:
            duplicates = []
            print(idx)
            for i, row in enumerate(idx):
                for j, elem in enumerate(row):
                    if float(dist[i][j]) < threshold:
                        duplicates.append((binx * batch_size + int(elem), i))
            return duplicates


def semantic_similarity(train_loader, test_loader, K, batch_size, model, print_dupes):
    emb_train = generate_embeddings(train_loader, model)
    emb_test = generate_embeddings(test_loader, model)
    duplicates = knn(emb_train, emb_test, K, batch_size, print_dupes)
    print(duplicates)
    return duplicates


if __name__ == '__main__':
    i = imagedataset()
    t = testdataset()
    newmodel = torchvision.models.resnet18(pretrained=True).to(device)
    newmodel.eval()

    batch_size = 1024
    K = 5
    train_data = torchvision.datasets.ImageFolder(root='/mnt/f/cpp/', transform=transforms_x)

    trainloader = torch.utils.data.DataLoader(i, batch_size=batch_size,
                                              shuffle=False, num_workers=12)

    testloader = torch.utils.data.DataLoader(t, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    folder = '/mnt/f/amp/a/'
    files = os.listdir(folder)
    files = [f"{folder}{f}" for f in files]
    simil = []
    print(files[14640])
    print(files[4500])

    before = time.time()
    semantic_similarity(trainloader,
                        testloader, K=5, batch_size=batch_size,
                        model=newmodel, print_dupes=True)
    print(time.time() - before)