# Did it spill?
Did you manage to spill samples from your train set to your test set?  


```python
from did_it_spill import check_spill

spills = check_spill(train_loader, test_loader)

print(f"You have {len(spills)} spills in your test set!")
```
The library computes hashes of your data to determine if you have samples spilled over from your train set to test set. 

## Installation
```
pip install did-it-spill
```
## Outputs
Function outputs a list of tuples. Each tuple corresponds to a leak. The first index is where in the first loader the 
leak was found, and the second index is the index where the the leak was found in the second loader.

Example output: 
```python
[(1244, 78)...(8774, 5431)]
```
The first leak was found at index 1244 in loader 1 and at index 78 in loader 2.



## Debugging spills
The unthinkable happen. So what should I do now?
```python
for spill in spills:
    # Lets get both of the samples and double check that they really are the same
    index_train_set = spill[0]
    index_test_set = spill[1]

    # Get the data from dataset
    spilled_sample_train = train_dataset.__getitem__(index_train_set)[0]
    spilled_sample_test = test_dataset.__getitem__(index_test_set)[0]
    
    # This should always be true
    print(torch.equal(spilled_sample_train, spilled_sample_test))
    
    # From here on its up to you, maybe plot the data?
    print(spilled_sample_train)
```