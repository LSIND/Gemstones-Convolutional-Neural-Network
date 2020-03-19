# Gemstones Neural Network - multiclass classification

Can your program understand is it Ruby, Amethyst or Emerald?

![Alexandrite](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/alexandrite-chrysoberyl-brazil-t.jpg&size=120)
![Aquamarine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/Aquamarine_trillion_cut-thb.jpg&size=120)
![Citrine](https://www.minerals.net/thumbnail.aspx?image=GemStoneImages/CITRINE-cushion-thb.jpg&size=120)

> ### Build a simple convolutional neural network (CNN) from scratch over the dataset of gemstones images.

> FULL DATASET OF GEMSTONES IMAGES CAN BE FOUND AT [MY KAGGLE PAGE](https://www.kaggle.com/lsind18/gemstones-images): it's already divided into train and test data. This dataset contains 3,000+ images of different gemstones.

* I. [Create dataset: parse images](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/1_Create_Dataset.md): from static and dynamic resources, create train and test folders (90%:10%)
* II. [Install libraries for building Neural Network](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/2_Install_Libraries.md) (+ GPU)
* III. [Prepare to build Convolutional Neural Network](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/3_Prepare_to_build_CNN.md): working over train folder (read and convert images and labels into numpy-arrays, split train data into train and validation sets)
* IV. [Build a Convolutional Neural Network](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/4_Build_CNN.md) for multiclass classification
        * import necessary modules
        * Provide Hyperparameters
        * Provide a model: choose an architecture (layers) and compile it; get the summary of the model
* V. [Fit the CNN](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/5_Fit_the_CNN.md): provide more data - image augmentation, fit the model with train data and wait `fit_generator` to complete its work.
* VI. [Undestand the model](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/6_Understand_CNN.md): plot accuracy and loss, score model, understand confusion matrix, save weights 
* VII [Evaluate on testing folder](https://github.com/LSIND/Gemstones-Neural-Network/blob/master/7_Evaluate_Model.md): understand new data, some conclusion.

*Feel free to give any suggestions to improve my code.*
