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
