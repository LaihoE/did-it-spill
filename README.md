# Did it spill?
Computes hashes of you data to determine if you have samples spilled over from train to test. Function also returns duplicates
inside from inside the same loader
```python
from did_it_spill import checkspill

dupes_loader_1, dupes_loader_2, test_spills = checkspill(train_loader, test_loader)

print(f"Loader 1 had {len(dupes_loader_1)} duplicates")
print(f"Loader 1 had {len(dupes_loader_2)} duplicates")
print(f"Loader 2 had {len(test_spills)} spills")
```
## Installation
```
pip install did-it-spill
```
