# VII. Evaluate on testing folder

## 1. Get samples from test folder
Create test data generator using class `ImageDataGenerator` and validate it providing the test directory `'/kaggle/input/gemstones-images/test/'`

Create `Test_Imgs` and `Test_Lbls` absolutely the same as we did with training folder. Convert them to numpy arrays - there are 358 images for test. `Test_Lbls` array will help to check is the model predictions are correct.


## 2. Plot test images and model predictions
Plot sample, class which model predicted, and actual class.

# Conclusion

The model tries! Finally it understands the color: some gemstones are really similar.  

![Diamonds, really?](https://cdn.leibish.com/media/mediabank/blue-diamond-scale_1634.c8fbb.jpg)

**Feel free to give any suggestions to improve my code.**
