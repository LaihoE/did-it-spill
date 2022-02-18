# Did it spill?
Computes hashes of you data to determine if you have samples spilled over from train set to test set. Function also returns duplicates
inside from inside the same loader
```python
from did_it_spill import check_spill

dupes_loader_1, dupes_loader_2, test_spills = check_spill(train_loader, test_loader)

print(f"Loader 1 had {len(dupes_loader_1)} duplicates")
print(f"Loader 1 had {len(dupes_loader_2)} duplicates")
print(f"Loader 2 had {len(test_spills)} spills")
```
## Installation
```
pip install did-it-spill
```
Dupes_loader_1 and Dupes_loader_2 are dictionaries with the hash as the key and a list of indexes as value.  

test_spills is a list of tuples where index 0 is the index in first loader and index 1 is index in loader 2
### You can also call this short version that quits if you have spills, else nothing happens
```python
from did_it_spill import check_spill_assert

check_spill_assert(train_loader, test_loader)
```
Main useage of the library would be to always put this one line in your training script and forget about it. If you end up
having spills then you can use the above to debug.