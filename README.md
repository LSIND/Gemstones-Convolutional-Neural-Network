# Gemstones Neural Network - multiclass classification

Can your program understand is it Ruby, Amethyst or smth else?

![Alexandrite](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/alexandrite-chrysoberyl-brazil-t.jpg&size=120)
![Aquamarine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/Aquamarine_trillion_cut-thb.jpg&size=120)
![Citrine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/CITRINE-cushion-thb.jpg&size=120)


## 1. Download gemstones images
Fetch data from different sources, f.e. minerals.net

Install packages:  
`pip install bs4`  
`pip install requests`  

*Beautiful Soup (bs4)* = Python library for pulling data out of HTML and XML files.
*requests* = HTTP library

### [Example of downloading static content](fetch_data.py):
Scraping [minerals.net](https://www.minerals.net) website:
1. Get the page with [list of all gemstones](https://www.minerals.net/GemStoneMain.aspx) and find HTML-element with them:
```python
url = 'https://www.minerals.net/GemStoneMain.aspx'
html = requests.get(url).text
soup = bs(html, 'html.parser')
table_gems=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
```
3. Parse links to the pages of each gemstone and create dictionary {gemstone name : link }

4. Parse each page to get pictures of gemstones
```python
table_images=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
```

### [Example of downloading dynamic content](fetch_data2.py) using `selenium`:

`from selenium import webdriver`


## 2. Create Train and Test sets

`pip install split_folders`   
Split all images into train and test data
```python
split_folders.ratio('input', output="data", seed=1337, ratio=(.8, .2))
```
Check number of files in folders:
```diff
\data\train\Serpentine: filecount: 27
\data\train\Sodalite: filecount: 7
...
\data\val\Amber: filecount: 3
```

FULL DATASET OF GEMSTONES IMAGES CAN BE FOUND AT [MY KAGGLE PAGE](https://www.kaggle.com/lsind18/gemstones-images): it's already devided by train and test data.  
For each class, there are about 10-50 photos. Photos are about 300 x 300 pixels and are not reduced to a single size.

## 2. Install libraries

for building the simpliest neural network we need to install:
`pip install tensorflow`  
`pip install keras`  

## *install CUDA (optional)*
If you want to train your neural network on GPU check first your [GPU capability](https://developer.nvidia.com/cuda-gpus).   
*I tested on nvidia geforce gtx 1060 6Gb*   
Install the [NVIDIA® CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit) for running scripts on a GPU-accelerated system and the [NVIDIA CUDA® Deep Neural Network library (cuDNN)](https://developer.nvidia.com/cudnn) which is a GPU-accelerated library of primitives for deep neural networks.  

Enable GPU support for tensorflow:  
`pip install tensorflow-gpu`  


## 3. Image augmentation
Image augmentation is a creation of additional training data based on existing images, for example translation, rotation, flips and zoom.
Using Keras [create 20 additional images of each gemstone class](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/3_images_augm.py). Files are written to disk! You can write them to memory.

```python
img_gen = ImageDataGenerator(  
        rotation_range=40,   
        width_shift_range=0.2,   
        height_shift_range=0.2,    
        shear_range=0.2,    
        zoom_range=0.2,    
        horizontal_flip=True,    
        fill_mode='nearest')
```

## 4. Build simple CNN
CNN (Convolutional neural network or ConvNet) is a class of deep neural networks, commonly applied to analyzing visual imagery. Here is the simpliest example of CNN with a few layers.
```python
from keras.models import Sequential
```
Hyperparameters are as follows:
- 

