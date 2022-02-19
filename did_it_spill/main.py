import numpy as np
from collections import Counter
import hashlib


def get_dupes_inx(hashes_loader):
    dupecounts = Counter(hashes_loader)
    dupes_loader = {}
    for k, v in dupecounts.items():
        # If more than one time = duplicate
        if v > 1:
            dupes = []
            # Iterate over all hashes in loader, if match then add to outgoing list of indexes
            for i in range(len(hashes_loader)):
                if k == hashes_loader[i]:
                    dupes.append(i)
            dupes_loader[k] = dupes
    return dupes_loader


def check_spill(loader1, loader2):
    hash_loader1, hash_loader2 = [], []     # All hashes
    hasd_1, hasd_2 = {}, {}     # key=hash, val=index
    test_spills = []
    for loader_inx, loader in enumerate([loader1, loader2]):
        # Index in loader (What index in the dataset we are at)
        index = 0
        for data, _ in loader:
            # Make sure the data is the correct type (maybe unnecessary), causes problems without this
            data = data.detach().numpy()
            for i in range(data.shape[0]):
                index += 1
                d = data[i].view(np.uint8)
                this_hash = hashlib.sha1(d).hexdigest()
                if loader_inx == 0:
                    hash_loader1.append(this_hash)
                    hasd_1[this_hash] = index
                else:
                    hash_loader2.append(this_hash)
                    hasd_2[this_hash] = index

    # Unique hashes
    set_hash_loader1 = set(hash_loader1)
    set_hash_loader2 = set(hash_loader2)
    # Dupes from loaders
    dupes_loader_1 = get_dupes_inx(hash_loader1)
    dupes_loader_2 = get_dupes_inx(hash_loader2)
    # find spilled samples
    for hsh in set_hash_loader1:
        if hsh in set_hash_loader2:
            test_spills.append((hasd_1[hsh], hasd_2[hsh]))
    return dupes_loader_1, dupes_loader_2, test_spills


def check_spill_assert(loader1, loader2):
    dupes_loader_1, dupes_loader_2, test_spills = check_spill(loader1, loader2)
    assert len(test_spills) == 0, f'test set has {len(dupes_loader_1)} spills !'
    assert len(dupes_loader_1) == 0, f'Loader 1 had {len(dupes_loader_1)}'
    assert len(dupes_loader_2) == 0, f'Loader 2 had {len(dupes_loader_1)}'




