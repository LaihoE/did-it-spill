from typing import List, Tuple, Dict
import numpy as np
import hashlib
from torch.utils.data import DataLoader


def __get_hash_dict(loader: DataLoader) -> Dict[str, int]:
    index = 0
    hash_dict = {}
    # For each batch
    for data, _ in loader:
        data = data.detach().numpy()
        # For each sample in the batch
        for sample in range(data.shape[0]):
            # Hash the current sample
            this_hash = hashlib.sha1(data[sample].view(np.uint8)).hexdigest()
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
