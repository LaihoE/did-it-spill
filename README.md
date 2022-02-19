# Did it spill?
Did you manage to spill samples from your train set to your test set?  


```python
from did_it_spill import check_spill

dupes_loader_1, dupes_loader_2, test_spills = check_spill(train_loader, test_loader)

print(f"Loader 1 had {len(dupes_loader_1)} duplicates")
print(f"Loader 2 had {len(dupes_loader_2)} duplicates")

print(f"You have {len(test_spills)} spills in your test set!")
```
The library computes hashes of your data to determine if you have samples spilled over from train set to test set. Function also returns duplicates
 from inside the same loader.
Currently only for PyTorch  
### You can also call this short version that quits if you have spills, else nothing happens
```python
from did_it_spill import check_spill_assert

check_spill_assert(train_loader, test_loader)
```

## Installation
```
pip install did-it-spill
```
## Debugging spills

Example output of test_spills: ```[(32247, 187)...]```  
Here the first spill was found in the training set at index 32247 and at index 187 in the test set (assuming loader 1 was training loader).

Example output of dupes: ```{867fe5da1f164ecc5159d99fb46cc893ea1a3d44: [202, 203, 204], ...}```
Here the sample corresponding to the hash "867fe5da1f164ecc5159d99fb46cc893ea1a3d44" was found 3 times: at indexes 202, 203 and 204.