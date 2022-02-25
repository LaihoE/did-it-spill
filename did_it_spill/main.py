import numpy as np
import hashlib


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
                d = data[i].view(np.uint8)
                this_hash = hashlib.sha1(d).hexdigest()
                if loader_inx == 0:
                    hash_loader1.append(this_hash)
                    hasd_1[this_hash] = index
                else:
                    hash_loader2.append(this_hash)
                    hasd_2[this_hash] = index
                index += 1
    # Unique hashes
    set_hash_loader1 = set(hash_loader1)
    set_hash_loader2 = set(hash_loader2)
    # find spilled samples
    for hsh in set_hash_loader1:
        if hsh in set_hash_loader2:
            test_spills.append((hasd_1[hsh], hasd_2[hsh]))
    return test_spills
