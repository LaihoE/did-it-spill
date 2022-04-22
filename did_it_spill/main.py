from typing import List, Tuple, Dict
import numpy as np
import hashlib
from torch.utils.data import DataLoader, Dataset


def __get_hash_dict(loader: DataLoader) -> Dict[str, int]:
    index = 0
    hash_dict = {}
    # For each batch
    for data, _ in loader:
        data = data.detach().numpy()
        # For each sample in the batch
        for sample in data:
            # Hash the current sample
            this_hash = hashlib.sha1(sample).hexdigest()
            # Key = hash of our sample, val = where in the dataset that hash was
            hash_dict[this_hash] = index
            index += 1
    return hash_dict


def check_spill(loader1: DataLoader, loader2: DataLoader) -> List[Tuple[int, int]]:
    test_spills = []
    hashes_loader1 = __get_hash_dict(loader1)
    hashes_loader2 = __get_hash_dict(loader2)
    # find spilled samples
    for hsh in hashes_loader1.keys():
        if hsh in hashes_loader2.keys():
            test_spills.append((hashes_loader1[hsh], hashes_loader2[hsh]))
    return test_spills


def get_spilled_samples(spills: List, train_dataset: Dataset):
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
