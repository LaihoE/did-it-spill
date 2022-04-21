import time
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import os
import glob
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


def generate_embeddings(loader, model=None):
    """
    Generates embeddings from images by simply doing a forward pass.
    This seems to work unreasonably well.
    The embedding is simply the last layer of the model. By default using
    models trained on imagenet this will be dim=1000. Notice that using
    normal CNN trained on image classification might not be SOTA for
    this, but it seems to work darn well!

    By default it uses a resnet18, but the user can also specify a custom model.
    Function will use the output of specified custom model as the embedding.

    This is also the function that takes over 99 % of the time.
    :param loader:
    :param model:
    :return:
    """
    if model is None:
        model = torchvision.models.resnet18(pretrained=True).to(device)
    model.eval()
    # returns embeddings in shape (n, 1000)
    embeddings = []
    for data in loader:
        embedding = model(data.to(device))
        embeddings.append(embedding.flatten(start_dim=1).detach().cpu())
    return torch.cat(embeddings)


def knn(emb_train, emb_test, K, batch_size, threshold=30, print_dupes=False):
    """
    KNN that supports GPU. VERY fast.

    :param emb_train:
    :param emb_test:
    :param K:
    :param batch_size:
    :param threshold:
    :param print_dupes:
    :return:
    """
    for binx in range(math.ceil(len(emb_test) / batch_size)):
        # Slice the test set
        batch_test_emb = emb_test[binx * batch_size: binx * batch_size + batch_size, :]
        # Compute dist between test embeddings and train embeddings
        D = torch.cdist(batch_test_emb, emb_train)
        # Get distances and indexes to top k closest vectors
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


def semantic_similarity(train_loader, test_loader, K, batch_size, print_dupes, model=None):
    """
    Returns indexes for images that are very similar or identical.

    Generates embeddings by doing a forward pass with a pretrained resnet18 (by default).
    Alternatively you can specify a custom model for generating embeddings.
    The embeddings are then compared with KNN to find similar images.

    :param train_loader:
    :param test_loader:
    :param K:
    :param batch_size:
    :param model:
    :param print_dupes:
    :return:
    """
    emb_train = generate_embeddings(train_loader, model)
    emb_test = generate_embeddings(test_loader, model)
    duplicates = knn(emb_train, emb_test, K, batch_size, print_dupes)
    return duplicates

class Imagedataset(Dataset):
    def __init__(self, dir, recursive=False):

        if recursive:
            self.files = []
            for filename in glob.iglob(f'{dir}/**', recursive=True):
                if os.path.isfile(filename):
                    self.files.append(filename)
        else:
            self.files = os.listdir(dir)
            self.files = [f"{folder}{f}" for f in self.files]
        self.n_samples = len(self.files)

        self.transforms = torch.nn.Sequential(
            T.Resize((64, 64)),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )

    def __getitem__(self, index):
        raw_image = read_image(self.files[index])
        transformed_image = self.transforms(raw_image)
        return transformed_image

    def __len__(self):
        return self.n_samples


def dupes_from_folder(dir, K, batch_size, recursive=False, model=None, num_workers=0, print_dupes=False):
    """
    Returns file paths of images that are near duplicates or identical.
    Example output [(img1.jpg, img2.jpg) ...]

    :param dir:
    :param K:
    :param batch_size:
    :param recursive:
    :param model:
    :param num_workers:
    :param print_dupes:
    :return:
    """
    duplicate_files = []
    dataset = Imagedataset(dir, recursive)
    files = dataset.files
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)

    embeddings = generate_embeddings(data_loader, model)
    duplicates = knn(embeddings, embeddings, K, batch_size)
    for dupe in duplicates:
        if dupe[0] != dupe[1]:
            image1 = files[dupe[0]]
            image2 = files[dupe[1]]
            if print_dupes:
                print(image1, "-->", image2)
            duplicate_files.append((image1, image2))
    return duplicate_files


def get_spilled_samples(spills, train_dataset):
    """
    Returns the actual data that was spilled. Notice that it
    returns everything that the __getitem__ returns ie. data and labels
    and potentially other stuff. This is done to be more
    general, not just work with datasets that return: (data, label),
    but also for datasets with (data, label, third_thing) or similar.

    Notice that the function only takes in one dataset but spill
    is a tuple with indexes for two datasets (the other is ignored).
    :param spills:
    :param train_dataset:
    :return: spilled_samples:
    """
    spilled_samples = []
    for spill in spills:
        spill_inx = spill[0]
        spilled_samples.append(train_dataset.__getitem__(spill_inx))
    return spilled_samples


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


    before = time.time()
    semantic_similarity(trainloader,
                        testloader, K=5, batch_size=batch_size,
                        model=newmodel, print_dupes=True)
    print(time.time() - before)