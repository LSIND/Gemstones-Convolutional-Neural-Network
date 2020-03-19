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
