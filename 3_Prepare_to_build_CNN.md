# III. Prepare to for build CNN

## 1. Prepare training data

1. Prepare parameters
* resize all images to `img_w, img_h` - this option will be used as a parameter of neural network 
* provide train directory path
2. Create function to read images into lists
* this function will be also used with test images;
* read each image from disk using `cv2` and resize it to `img_w*1.5, img_h*1.5`;
* set `cv2.COLOR_BGR2RGB` option because opencv reads and displays an image as BGR color format instead of RGB color format. Without this option images will be shown in blue hue because `matplotlib` uses RGB to display image;
* create a list of class names while reading folders - `Amethyst, Onyx, etc`;
* when `Images` list is ready - convert it to Numpy array;
* return tuple of 2 elements: Images and corresponding Labels.

```Python
def read_imgs_lbls(_dir):
    Images, Labels = [], []
    for root, dirs, files in os.walk(_dir):
        f = os.path.basename(root)                             # get class name - Amethyst, Onyx, etc       
        for file in files:
            Labels.append(f)
            try:
                image = cv2.imread(root+'/'+file)              # read the image (OpenCV)
                image = cv2.resize(image,(int(img_w*1.5), int(img_h*1.5)))  # resize the image (images are different sizes)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)              # converts an image from BGR color space to RGB
                Images.append(image)
            except Exception as e:
                print(e)
    Images = np.array(Images)
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
    Labels = np.array(Labels)
    return Labels
```
4. Fill numpy arrays of Images and corresponding Labels with data
* Create two arrays `Train_Imgs, Train_Lbls` which contain images and corresponding names of classes of gemstones respectively;
* Convert `Train_Lbls` with strings to list with corresponding numbers;
* print the dimensions of both numpy arrays: `Train_Imgs` which stores pictures is 4-dimensional: **Number of images x Width of image x Height of image x Channel of image**.

```Python
Train_Imgs, Train_Lbls = read_imgs_lbls(train_dir)
Train_Lbls = get_class_index(Train_Lbls)
```

5. Plot images and their labels for preview
* Using `matplotlib` and `random` show 16 (4x4) random images from the set and their labels (as string and as number).
![train example](https://www.dropbox.com/s/wog6y095a6efqv2/train1.JPG?raw=1)

6. Crop edges of images using Canny algorithm
> Canny is a popular edge detection algorithm, which detects the edges of objects present in an image.

* Using `cv2.Canny` find the array representing frame which is the edges how the original picture will be cut;
* Function `edge_and_cut(img)` receives single image and returns a **cropped image (`new_img`)** of the size `img_w, img_h`;
* sometimes Canny algo cannot detect edges (f.e. when the object has almost same color as background) so array `edges` will be zero-valued. In this case use original image.
![canny edges ex](https://www.dropbox.com/s/xevmf3sdsxexftf/canny_edges_ex.JPG?raw=1)

7. Replace train images with cropped images
* Create function which replaces `Train_Imgs` numpy array with array of cropped images. Don't forget that images that cannot be cropped will be replaced with originals;
* Make sure the shape of final array is the same: NUMBER OF IMAGES x img_w x img_h x 3 (CHANNELS)

## 2. Split train data into train and validation sets
* use `sklearn` to split `Train_Imgs`, `Train_Lbls` into train (80%) and validation (20%) sets. **Important!**

```python
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(Train_Imgs, Train_Lbls, 
                                                shuffle = True, test_size = 0.2, random_state = 42)
```
