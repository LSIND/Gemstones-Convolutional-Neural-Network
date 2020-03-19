# VI. Understand the accuracy of the model

## 1. Check the accuracy

Plot the accuracy of model against size of epoch (train and val)  
Plot the loss of model against size of epoch (train and val)

```python
...
axs[0].plot(m.history['accuracy'])
axs[0].plot(m.history['val_accuracy'])
axs[1].plot(m.history['loss'])
axs[1].plot(m.history['val_loss'])
```

-->>>>>>>>>>>>>>>>>>graph

* accuracy keeps increasing: probably providing more epochs can improve a model
* there is a some overfitting: even though train and val accuracy are pretty close to each other, sometimes val_loss 'jumps'

## 2. Score the model

Function `evaluate_generator` evaluates the model on a data generator.  
In this case score is a list of scalars (loss and accuracy).

```
loss:0.8221790790557861
accuracy:0.6345810890197754
```

So the accuracy for a model from scratch ~ 65%. Any suggestions on improving a model are taking up 😊

## 3. Confusion matrix   
Confusion matrix can be pretty useful when evaluating multiclass classifications. Because of great amount of classes just **plot misclassified gemstones by model**. `numpy.argmax()` function returns the indices of maximum elements along the specific axis inside the array (`axis = 1` - 'horizontally').

Create a list of misclassified indexes which will be substitued into validation set `X_val`.  
Plot misclassified gemstones.
```
x=(y_pre_test-y_val!=0).tolist()
x=[i for i,l in enumerate(x) if l!=False]
```

Don't judge poor model. Just look at `Almandine`, `Garnet Red`, `Hessonite`, `Pyrope` and `Rhodolite`. Can you distinguish between them?
-->>>> IMAGES

## 4. Save the model
* Save weights to reuse them instead of training again. Keras function `save` creates h5 file with weights. Use `new_model.load_weights('model_gemstones.h5')` to reuse it in other models.
```python
model.save('model_gemstones.h5')
```
