import numpy as np
from typing import List, Tuple, Dict
from numpy.typing import NDArray
import hashlib

# Not done yet

def __get_hash_dict(data: NDArray) -> Dict[str, int]:
    index = 0
    hash_dict = {}
    for sample in data:
        # Hash the current sample
        this_hash = hashlib.sha1(sample).hexdigest()
        # Key = hash of our sample, val = where in the dataset that hash was
        hash_dict[this_hash] = index
        index += 1
    return hash_dict


def check_spill(train: NDArray, test: NDArray) -> List[Tuple[int, int]]:
    test_spills = []
    hashes_loader1 = __get_hash_dict(train)
    hashes_loader2 = __get_hash_dict(test)
    # find spilled samples
    for hsh in hashes_loader1.keys():
        if hsh in hashes_loader2.keys():
            test_spills.append((hashes_loader1[hsh], hashes_loader2[hsh]))
    return test_spills
