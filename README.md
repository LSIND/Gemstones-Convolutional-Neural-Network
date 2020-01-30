# Gemstones Neural Network - multiclass classification

Can your program understand is it Ruby, Amethyst or Emerald?

![Alexandrite](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/alexandrite-chrysoberyl-brazil-t.jpg&size=120)
![Aquamarine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/Aquamarine_trillion_cut-thb.jpg&size=120)
![Citrine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/CITRINE-cushion-thb.jpg&size=120)

> ### Build a simple convolutional neural network (CNN) from scratch over the dataset of gemstones images.

# I. Create Gemstones Dataset
> FULL DATASET OF GEMSTONES IMAGES CAN BE FOUND AT [MY KAGGLE PAGE](https://www.kaggle.com/lsind18/gemstones-images): it's already divided into train and test data. This dataset contains 3,000+ images of different gemstones.

## 1. Download gemstones images
Example of fetching data from different sources in 2 ways: scraping static content and scraping dunamic content.

Install packages:  
```Console
pip install bs4 
pip install requests
pip install selenium
```
*Beautiful Soup (bs4)* = Python library for pulling data out of HTML and XML files.  
*requests* = HTTP library  
*selenium* = bindings for Selenium WebDriver; automate web browser interaction from Python.  

### [Parse static content](1_Fetch_data/fetch_data.py):
Example of scraping [minerals.net](https://www.minerals.net) website:
1. Get the page with [list of all gemstones](https://www.minerals.net/GemStoneMain.aspx) and find HTML-element with them:
```python
url = 'https://www.minerals.net/GemStoneMain.aspx'
html = requests.get(url).text
soup = bs(html, 'html.parser')
table_gems=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
```
2. Parse links to the pages of each gemstone and create dictionary `{gemstone name : link }`

3. Parse each page to get pictures of gemstones
```python
table_images=soup.find_all('table',{'id':'ctl00_ContentPlaceHolder1_DataList1'})
```

### [Parse dynamic content](1_Fetch_data/fetch_dyn_data.py) using `selenium`:
Example of scraping [www.rasavgems.com](https://www.rasavgems.com) website. The website uses javascript.
1. Import webdriver
```python
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
```
2. Download web driver for you browser and include it into code 
```python
wd = webdriver.Chrome("....\\chromedriver_win32\\chromedriver.exe") 
```
3. Get data using automatic interaction
```python
element = wait.until(
        EC.presence_of_element_located((By.ID, "Product_List")))
        imgs_form = wd.find_element_by_id('Product_List')
```

We created one folder with subfolders with gemstones pictures inside. They can be checked manually: finally I got 87 classes of gemstones.

## 2. Create a dataset for NN: Train and Test sets

### [Rename images](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/1_Fetch_data/2_Rename_Files.py) in Train and Test sets

1. Using `os` module rename files in every folder with name `FolderName_N.extension`. For example, for Ametrine folder rename files as `ametrine_0.jpg`, `ametrine_1.jpg` etc.

2. Avoid common problems: folder iss empty or file with such name already exists.
```Console
...
input\\Chalcedony
CCOV1520CSCABIOCCMLT-1003_1.jpg                     --> chalcedony_0.jpg
chalcedony-round-multi-color-ccmlt500x500.jpg       --> chalcedony_1.jpg
....
input\\Labradorite
labradorite-gem-253954a.jpg                         --> labradorite_0.jpg
...
```

### [Split images](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/1_Fetch_data/2_Split_to_Train_Test.py) into train and test data: 90% : 10%
> Train set is used to teach a neural network. Test set is used to check if the neural network understand a gemstone or not.

1. Use [split_folders](https://pypi.org/project/split-folders/)  
`pip install split_folders`   

This library provides splitting folders with files into train, validation and test (dataset) folders. Split with a ratio to only split into training and validation set `(.9, .1)`.

```python
split_folders.ratio('input', output="data", seed=1337, ratio=(.9, .1))
```
2. [Check number of files and folders](https://github.com/LSIND/intro-to-python3-analysis/tree/master/CountFilesAndFolders/main.py):
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


# II. Install libraries for building NN

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
Run the code to train CNN with GPU mode: every epoch on CPU takes ~3 minutes, on GPU ~ 15 sec.

```Python
from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
print(devices)
```

# III. Preparation steps for building CNN

## 1. Prepare training data

1.Prepare parameters
* resize all images to `img_w, img_h` - this option will be used as a parameter of neural network 
* provide train directory path
2. Create function to read images into lists
* this function will be also used with test images
* read each image from disk
* import OpenCV (`import cv2`): an image and video processing library  
* set `COLOR_BGR2RGB` option because opencv reads and displays an image as BGR color format instead of RGB color format. Without this option images will be shown in blue hue because `matplotlib` uses RGB to display image

```Python
def read_imgs_lbls(_dir):
    Images, Labels = [], []
    for root, dirs, files in os.walk(_dir):
        path = root.split(os.sep)
        f = os.path.basename(root)
        for file in files:
            Labels.append(f)
            try:
                image = cv2.imread(root+'/'+file)              # read the image (OpenCV)
                image = cv2.resize(image,(img_w, img_h))       # resize the image (images are different sizes)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # converts an image from BGR color space to RGB
                Images.append(image)
            except Exception as e:
                print(str(e))
    return (Images, Labels)
```

3. Create function to convert string labels to numbers
* Convert labels to a list of numbers not words using list `CLASSES`. The index will represent label of class, f.e. *Ruby = 0, Amethyst = 24*, etc.

```Python
def get_class_index(Labels):
    for i, n in enumerate(Labels):
        for j, k in enumerate(CLASSES):    # foreach CLASSES
            if n == k:
                Labels[i] = j
    return Labels
```
4. Fill lists of images and labels with data
* create two lists `Train_Imgs, Train_Lbls = [], []`. `Train_Imgs` list contains `cv2` images and `Train_Lbls` contains classes' names of gemstones.
* 'Convert' `Train_Lbls` with strings to list with corresponding numbers
5. Convert lists of images and labels to numpy arrays

Before creating model we need to convert lists of images and labels to numpy arrays to feed them to it.
* Create function for this routine work  
Pass `Train_Imgs, Train_Lbls` to the function `lists_to_np_arr` to convert them to Numpy arrays. Array `Train_Imgs` with images is 4-dimensional: **Number of images x Width of image x Height of image x Channel of image**.
6. Plot images
* Using `matplotlib` and `random` show 16 (4x4) random images from the set and their labels (as string and as number).

## 2. Split train data into train and validation sets
* use `sklearn` to split `Train_Imgs`, `Train_Lbls` into train (80%) and validation (20%) sets. **Important!**


# IV. Build a Convolutional Neural Network
> for multiclass classification

## 1. Import keras
Keras is an open-source neural-network library written in Python which is capable of running on top of **TensorFlow**.
From Keras needed:
* `models` - type of models, import only `Sequential` 
* `layers` - layers corresponding to our model: as it a simple one take only `Conv2D`, `MaxPooling2D` and `AveragePooling2D`
* `optimizers` - contains back propagation algorithms
* `ImageDataGenerator` - for image augmenation (there are not so many samples of each class)

## 2. Build a CNN  
CNN (Convolutional neural network or ConvNet) is a class of deep neural networks, commonly applied to analyzing visual imagery. Here is the simpliest example of CNN with few layers using `Conv2D` - 2D convolution layer (spatial convolution over images) and `MaxPooling2D` - application of a moving window across a 2D input space.

### 1. Provide Hyperparameters
Hyperparameters are set before training; they represent the variables which determines the neural network structure and how the it is trained.   
1. Parameters for layers
* Convolutional layer filter size (`filters`). The number of filters should depend on the complexity of dataset and the depth of neural network. A common setting to start with is [32, 64, 128] for three layers.  
* `kernel_size` = number of filters  = a small window of pixels at a time (3×3) which will be moved until the entire image is scanned. If images are smaller than 128×128, work with smaller filters of 1×1 and 3×3
* Width and Height of images were already provided. 2D convolutional layers take a three-dimensional input, typically an image with three color channels
* `max_pool` = max pooling is the application of a moving window across a 2D input space, where the maximum value within that window is the output: 2x2 
2. Parameters to fit the model
* **epoch** describes the number of times the algorithm sees the ENTIRE dataset. Each time the algo has seen all samples in the dataset, an epoch has completed.  
* since one epoch is too big to feed to the memory at once divide it in several smaller **batches**. Batch size is always factor of 2. 
* **Iterations** per epoch = number of passes, each pass using batch size number of examples.   

So if we have ~2200 (80%) training samples, and batch size is 32, then it will take ~70 iterations to complete 1 epoch.

### 2. Provide a model

**Architect a model**
The Sequential model is a linear stack of layers.
* I use a kind of VGG network architecture:

|   |       Layers       |
|:-:|:------------------:|
| 1 |  Conv2D 32 -> Pool |
| 2 |  Conv2D 64 -> Pool |
| 3 | Conv2D 128 -> Pool |
| 4 | Conv2D 128 -> Pool |
| 5 | Conv2D 128 -> Pool |
| 6 |        FLAT        |
| 7 |        Drop        |
| 8 |      Dense 256     |
| 9 | Dense len(CLASSES) |


1. ADD 5 'blocks': 
   *  Conv2D with hypermarameters mantioned above: `Conv2D(kernel_size, (filters, filters), input_shape=(img_w, img_h, 3))` with activation function for each layer as a Rectified Linear Unit (ReLU): `Activation('relu')`  
   * MaxPooling2D layer to reduce the spatial size of the incoming features; 2D input space: `MaxPooling2D(pool_size=(max_pool, max_pool))`  
   * Do the same increading the kernel size: 32 -> 64 -> 128 -> 128 -> 128

2. Flatten the input: transform the multidimensional vector into a single dimensional vector: `Flatten()`
3. Add dropout layer which randomly sets a certain fraction of its input to 0 and helps to reduce overfitting: `Dropout(0.5)`
5. Add fully connected layer with 256 nodes and activation function relu: `Dense(256), Activation('relu')`
6. Provide last fully connected layer which specifies the number of classes of gemstones: **87**. `Softmax` activation function outputs a vector that represents the probability distributions of a list of potential outcomes: `Dense(87, activation='softmax')`   

* Print the summary of the model.
The model summary shows that there are more than 1,500,000 parameters to train and the information about different layers.

### 3. Compile a model
* Compile the model using `adam` optimizer which is a generalization of stochastic gradient descent (SGD) algo. Provided loss function is `sparse_categorical_crossentropy` as we are doing multiclass classification.   

## 3. Fit the train generator

### 1. Image augmentation

As far as there is not so much samples for every class add a train data generator using class `ImageDataGenerator` with augmentation parameters. **Image augmentation** is a creation of additional training data based on existing images, for example translation, rotation, flips and zoom. Using `ImageDataGenerator` class from Keras library create additional images of each gemstone class in the memory.

Create two numpy array iterators `train_gen` and `val_gen` and fill them with additional images:

### 2. Fit the model
* get a history object
* If you see that `val_los` parameter is increasing that is *overfitting*. It happens when your model explains the training data too well, rather than picking up patterns that can help generalize over unseen data.

😈 **ALMOST DONE!** 😈   

## 4. Check the accuracy

* plot the accuracy of model against size of epoch (train and val)
* plot the loss of model against size of epoch (train and val)

* accuracy keeps increasing: probably providing more epochs can improve a model
* there is a some overfitting: even though train and val accuracy are pretty close to each other, sometimes val_loss 'jumps'

## 5. Score the model

Function `evaluate_generator` evaluates the model on a data generator.  
In this case score is a list of scalars (loss and accuracy).

So the accuracy for a model from scratch ~ 65%. Any suggestions on improving a model are taking up 😊

## 6. Confusion matrix   
Confusion matrix can be pretty useful when evaluating multiclass classifications. Because of great amount of classes just **plot misclassified gemstones by model**. `numpy.argmax()` function returns the indices of maximum elements along the specific axis inside the array (`axis = 1` - 'horizontally').

Create a list of misclassified indexes which will be substitued into validation set `X_val`.  
Plot misclassified gemstones.

Don't judge poor model. Just look at `Almandine`, `Garnet Red`, `Hessonite`, `Pyrope` and `Rhodolite`. Can you distinguish between them?

## 7. Save the model
* Save weights to reuse them instead of training again. Keras function `save` creates h5 file with weights. Use `new_model.load_weights('model_gemstones.h5')` to reuse it in other models.

# V. Evaluate on testing folder

## 1. Get samples from test folder
Create test data generator using class `ImageDataGenerator` and validate it providing the test directory `'/kaggle/input/gemstones-images/test/'`

Create `Test_Imgs` and `Test_Lbls` absolutely the same as we did with training folder. Convert them to numpy arrays - there are 358 images for test. `Test_Lbls` array will help to check is the model predictions are correct.


## 2. Plot test images and model predictions
Plot sample, class which model predicted, and actual class.

# Conclusion

The model tries! Finally it understands the color: some gemstones are really similar.  

![Diamonds, really?](https://cdn.leibish.com/media/mediabank/blue-diamond-scale_1634.c8fbb.jpg)

**Feel free to give any suggestions to improve my code.**
