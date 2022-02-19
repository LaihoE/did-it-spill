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
## Outputs
**test_spills** = ```{72d2e3e611344c4a794b8e0e4ad520b1c36f918d: [32247, 187], ...}```  
index 0 = spill found in first loader at this index.  
index 1 = spill found in second loader at this index.  
Always len = 2

**dupes_loader_1 and 2** = ```{867fe5da1f164ecc5159d99fb46cc893ea1a3d44: [202, 203, 204], ...}```  
Duplicates found at these indexes  
Varying length

## Debugging spills
The unthinkable happen. So what should I do now?
```python
for _, v in test_spills.items():
    # Lets get both of the samples and double check that they really are the same
    index_train_set = v[0]
    index_test_set = v[1]

    # Get the data from dataset
    spilled_sample_train = train_dataset.__getitem__(index_train_set)[0]
    spilled_sample_test = test_dataset.__getitem__(index_test_set)[0]
    
    # This should always be true
    print(torch.equal(spilled_sample_train, spilled_sample_test))
    
    # From here on its up to you, maybe plot the data?
    print(spilled_sample_train)
```