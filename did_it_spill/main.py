from typing import List, Tuple, Dict

import numpy as np
import hashlib
from torch.utils.data import DataLoader


def check_spill(train_dataloader: DataLoader, test_dataloader: DataLoader) -> List[Tuple[int, int]]:
    """
    Check train- and test-dataloader for mixed samples
    :param train_dataloader: holds train-data
    :param test_dataloader: holds test-data
    :return: a list consisting of index-tuples which indicate the position of the mixed samples
    """
    hasd_1 = __make_hash_dict_from(train_dataloader)
    hasd_2 = __make_hash_dict_from(test_dataloader)
    test_spills = __get_overlap_of_hashdicts(hasd_1, hasd_2)
    return test_spills


def __get_overlap_of_hashdicts(hashdict1: Dict[str, int], hashdict2: Dict[str, int]) -> List[Tuple[int, int]]:
    overlapping_indexes = list()
    for hsh in hashdict1.keys():
        if hsh in hashdict2:
            overlapping_indexes.append((hashdict1[hsh], hashdict2[hsh]))
    return overlapping_indexes


def __make_hash_dict_from(dataloader: DataLoader) -> Dict[str, int]:
    hash_dict = dict() # key=hash, val=index
    for batch_index, (batch_x, batch_y) in enumerate(dataloader):
        # conversion needed since the hash is not possible with int32
        batch_x = np.uint8(batch_x.to("cpu").detach().numpy()) # pull tensor to the cpu-side and detach from gradient
        for data_index, data_x in enumerate(batch_x):
            this_hash = hashlib.sha1(data_x).hexdigest()
            hash_dict[this_hash] = batch_index + data_index

    return hash_dict
