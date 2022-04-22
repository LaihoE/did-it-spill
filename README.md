# Did it spill?
Did you manage to spill samples from your train set to your test set?  


```python
from did_it_spill import check_spill

spills = check_spill(train_loader, test_loader)

print(f"You have {len(spills)} spills in your test set!")
```
The library computes hashes of your data to determine if you have samples spilled over from your train set to test set.
Currently only for PyTorch.

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
**Make sure you have shuffle = False for correct indexes.**  
get_spilled_samples returns the actual data that was spilled.

```python
spills = check_spill(train_loader, test_loader)
# Notice that we are passing the TRAIN DATASET (or the same dataset that the first loader is using in above func)
spilled_data = get_spilled_samples(spills, train_dataset)
```
Notice that the spilled_data includes everything that the get_item would return ie. data and labels and potentially more. 
This is mainly done to support any type of dataset, but it might also be useful to have labels to find the underlying problem.

### Example output: 
```python
[(data, label) ... ]
```

# Semantic similarity
Now also supports checking for similar or identical images.  
There are 2 main functions for this:  


### duplicate_images()
```python
>>> spills = duplicate_images(train_loader, test_loader, K, batch_size)
>>> spills
[(1244, 78)...(8774, 5431)]
```
### duplicate_images_dir()
```python
>>> similar_images = duplicate_images_dir(dir, K, batch_size)
>>> similar_images
[("img48.jpg", "img21.jpg") ...]
```
K stands for the max amount of duplicate images returned per image. "how many neighbours KNN will return".
Tradeoff between speed and number of images.


Both functions do similar things, and only have slightly different inputs and outputs. duplicate_images() works very similarly to
the main check_spill function by taking in both loaders and returns the indexes of spills (too similar images).  

duplicate_images_dir() on the other hand operates completely with
files by taking in a directory and returns the images that are similar. This one might be easier for debugging.