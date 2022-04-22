import math
import os
import glob
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
import torchvision.transforms as T
from torchvision.io import read_image


class Imagedataset(Dataset):
    def __init__(self, path: str):
        self.files = []
        for filename in glob.iglob(f'{path}/**', recursive=True):
            if os.path.isfile(filename):
                self.files.append(filename)
        self.n_samples = len(self.files)
        self.transforms = torch.nn.Sequential(
            T.Resize((64, 64)),
            T.ConvertImageDtype(torch.float),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        )

    def __getitem__(self, index):
        raw_image = read_image(self.files[index])
        transformed_image = self.transforms(raw_image)
        return transformed_image

    def __len__(self):
        return self.n_samples


def __generate_embeddings(loader: DataLoader, model=None):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = torchvision.models.resnet18(pretrained=True).to(device)
    model.eval()
    # returns embeddings in shape (n, 1000)
    embeddings = []
    for data in loader:
        embedding = model(data.to(device))
        embeddings.append(embedding.flatten(start_dim=1).detach().cpu())
    return torch.cat(embeddings)


def __knn(emb_train, emb_test, K: int, batch_size: int, threshold=30):
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
        duplicates = []
        for i, row in enumerate(idx):
            for j, elem in enumerate(row):
                if float(dist[i][j]) < threshold:
                    duplicates.append((binx * batch_size + int(elem), i))
        return duplicates


def duplicate_images(train_loader: DataLoader, test_loader: DataLoader,
                     K:int, batch_size:int, model=None) -> List[Tuple[int, int]]:
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
    emb_train = __generate_embeddings(train_loader, model)
    emb_test = __generate_embeddings(test_loader, model)
    duplicates = __knn(emb_train, emb_test, K, batch_size)
    return duplicates


def duplicate_images_dir(dir: str, K: int, batch_size: int,
                         model=None, num_workers: int = 0,
                         print_dupes: bool = False) -> List[Tuple[str, str]]:
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
    dataset = Imagedataset(dir)
    files = dataset.files
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)
    embeddings = __generate_embeddings(data_loader, model)
    duplicates = __knn(embeddings, embeddings, K, batch_size)
    for dupe in duplicates:
        if dupe[0] != dupe[1]:
            image1 = files[dupe[0]]
            image2 = files[dupe[1]]
            if print_dupes:
                print(image1, "-->", image2)
            duplicate_files.append((image1, image2))
    return duplicate_files
