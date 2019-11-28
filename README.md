# Gemstones Neural Network - multiclass classification

Can your program understand is it Ruby, Amethyst or smth else?

![Alexandrite](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/alexandrite-chrysoberyl-brazil-t.jpg&size=120)
![Aquamarine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/Aquamarine_trillion_cut-thb.jpg&size=120)
![Citrine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/CITRINE-cushion-thb.jpg&size=120)

# I. Fetch images

## 1. Download gemstones images
Fetch data from different sources, f.e. minerals.net

Install packages:  
`pip install bs4`  
`pip install requests`  

*Beautiful Soup (bs4)* = Python library for pulling data out of HTML and XML files.
*requests* = HTTP library

### [Parse static content](fetch_data.py):
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

### [Parse dynamic content](fetch_data2.py) using `selenium`:

`from selenium import webdriver`


## 2. Create Train and Test sets

`pip install split_folders`   
Split all images into train and test data: 90%:10%
```python
split_folders.ratio('input', output="data", seed=1337, ratio=(.9, .11))
```
Check number of [files and folders](https://github.com/LSIND/intro-to-python3-analysis/tree/master/CountFilesAndFolders):
```Console
------------> gemstones-images:  2 folders,  0 files
--------------> test :   87 folders,  0 files
----------------> Alexandrite :  0 folders,  4 files
----------------> Almandine :    0 folders,  4 files
...
...
--------------> train :  87 folders,  0 files
----------------> Alexandrite :  0 folders,  31 files
----------------> Almandine :    0 folders,  29 files
----------------> Amazonite :    0 folders,  28 files
```

**FULL DATASET OF GEMSTONES IMAGES CAN BE FOUND AT [MY KAGGLE PAGE](https://www.kaggle.com/lsind18/gemstones-images): it's already divided into train and test data. This dataset contains 3,000+ images of different gemstones.**  
There are 87 classes of different gemstones. The images are in various sizes and are in .jpg format. All gemstones have various shapes - round, oval, square, rectangle, heart.  
`Each class in train set contains 27 - 48 images, in test set - 4 - 6 images.`

![Train and Test](https://www.kaggleusercontent.com/kf/23910731/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..kaasYz84ovrb36KJbRNO3g.cSwAx07qbMYzv99o-Fxu71u0mJIEG0LHwhxFNnQfD13XmZ9yyN9a-EYJ3TpWgVOsQtfjxAr1td7nWzsQMqHJAso5wn-lMqAsnxAvtvUpO8zC7cIRXML_Pd729m5bmIPFfWjZFyOAkesNdSmu4kZn5WMPi4Pcgesy0WcUWvpTCbYkNAE9H5nv9Flh3WvbH5i4AHZVOSlFqfCzMVg8tNCPBA.WgdoSzArYqXTgPmP4-9w5g/__results___files/__results___1_1.png)

# II. Install libraries

## 1. Install Tensorflow + Keras
for building the neural network we need to install:
**TensorFlow** is an open-source symbolic math software library which is used for machine learning applications such as neural networks. **Keras** is an open-source neural-network library written in Python which is capable of running on top of TensorFlow. 
`pip install tensorflow`  
`pip install keras`  

## *2. Install CUDA (optional)*
If you want to train your neural network on GPU check first your [GPU capability](https://developer.nvidia.com/cuda-gpus).   
Install the [NVIDIA® CUDA® Toolkit](https://developer.nvidia.com/cuda-toolkit) for running scripts on a GPU-accelerated system and the [NVIDIA CUDA® Deep Neural Network library (cuDNN)](https://developer.nvidia.com/cudnn) which is a GPU-accelerated library of primitives for deep neural networks.  

Enable GPU support for tensorflow:  
`pip install tensorflow-gpu`  

## 3. Check devices
Using tensorflow check which devices are installed in you system:

XLA_CPU device: CPU  
XLA_GPU device: NVIDIA GTX-1060 6GB  
XLA stands for accelerated linear algebra. It's Tensorflow's relatively optimizing compiler that can further speed up ML models.

*I tested on nvidia geforce gtx 1060 6Gb*  
Run this notebook with GPU mode: every epoch on CPU takes ~3 minutes, on GPU ~ 15 sec.

```Python
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
print(devices)
```

# III. Import libraries

From Keras needed:
* `models` - type of models, import only `Sequential` 
* `layers` - layers corresponding to our model: as it a simple one take only `Conv2D`, `MaxPooling2D` and `AveragePooling2D`
* `optimizers` - contains back propagation algorithms
* `ImageDataGenerator` - for image augmenation (there are not so many samples of each class)

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
- Hyperparameters are set before training; they represent the variables which determines the neural network structure and how the it is trained.  
  
### 1. Parameters for layers
* Convolutional layer filter size (`filters`). The number of filters should depend on the complexity of dataset and the depth of neural network. A common setting to start with is [32, 64, 128] for three layers.  
* `kernel_size` = number of filters  = a small window of pixels at a time (3×3) which will be moved until the entire image is scanned. If images are smaller than 128×128, work with smaller filters of 1×1 and 3×3
* Width and Height of images were already provided. 2D convolutional layers take a three-dimensional input, typically an image with three color channels
* `max_pool` = max pooling is the application of a moving window across a 2D input space, where the maximum value within that window is the output: 2x2 

