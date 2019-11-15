# Gemstones Neural Network

## 1. Download gems images
Fetch data from different sources, f.e. minerals.net

`pip install bs4`  
`pip install requests`  
`pip install bs4`  

Example of downloading static content:

Example of downloading dynamic content using `selenium`:


## 2. Install libraries

## install CUDA (add)

## 2. Create Train and Test sets

Check number of files in folders


## 3. Image augmentation
Image augmentation is a creation of additional training data based on existing images, for example translation, rotation, flips and zoom.

``` img_gen = ImageDataGenerator(  
        rotation_range=40,   
        width_shift_range=0.2,   
        height_shift_range=0.2,    
        shear_range=0.2,    
        zoom_range=0.2,    
        horizontal_flip=True,    
        fill_mode='nearest')
```
        

## 4. Build simple CNN
