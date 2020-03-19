# IV. Fit the train generator

## 1. Image augmentation

As far as there are not so many samples for every class, add a train data generator using class `ImageDataGenerator` with augmentation parameters. **Image augmentation** is a creation of additional training data based on existing images, for example translation, rotation, flips and zoom. Using `ImageDataGenerator` class from Keras library create additional images of each gemstone class in the memory.

```python
train_datagen = ImageDataGenerator(              # this is the augmentation configuration used for training
        rescale=1./255,
    #preprocessing_function=preprocess_input,
        rotation_range=25,
        zoom_range=0.3,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
        )

val_datagen = ImageDataGenerator(                # for val/testing only rescaling function
    #preprocessing_function=preprocess_input,
    rescale=1./255
)
```

Create two numpy array iterators train_gen and val_gen and fill them with additional images:
```python
train_gen = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_gen = val_datagen.flow(X_val, y_val, batch_size=batch_size)
```

## 2. Fit the model
* get a history object
* If you see that `val_los` parameter is increasing that is *overfitting*. It happens when your model explains the training data too well, rather than picking up patterns that can help generalize over unseen data.
```python
m = model.fit_generator(
       train_gen,
       steps_per_epoch= iter_per_epoch,
       epochs=EPOCHS,
       validation_data = val_gen,
       validation_steps = val_per_epoch,
       verbose = 1 # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
       )
```

## 3. ..Wait a little
```
Epoch 1/35
70/70 [==============================] - 23s 323ms/step - loss: 4.3764 - accuracy: 0.0201 - val_loss: 4.1561 - val_accuracy: 0.0202
Epoch 2/35
70/70 [==============================] - 19s 266ms/step - loss: 3.9559 - accuracy: 0.0344 - val_loss: 3.7394 - val_accuracy: 0.0510
Epoch 3/35
...................
Epoch 33/35
70/70 [==============================] - 19s 275ms/step - loss: 1.1393 - accuracy: 0.6000 - val_loss: 1.5982 - val_accuracy: 0.5936
Epoch 34/35
70/70 [==============================] - 19s 267ms/step - loss: 1.0144 - accuracy: 0.6509 - val_loss: 1.0117 - val_accuracy: 0.6011
Epoch 35/35
70/70 [==============================] - 20s 291ms/step - loss: 1.0418 - accuracy: 0.6554 - val_loss: 1.4992 - val_accuracy: 0.6446
 ```
```
