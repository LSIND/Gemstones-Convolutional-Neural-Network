# III. Prepare to for build CNN

## 1. Prepare training data

1. Prepare parameters
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
