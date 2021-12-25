# Q1. Summary
In this project, I am exploring the CIFAR-10 data set. First I have extracted the data set. Then I have passed the dataset in the CNN pipeline and exploring different parameters of metrics and loss function. I have used different loss functions such as Jaccard, 
dice loss and dice coefficient. as we know that all these functions are for the segmentation for encoders and decoders. I still wanted to use these functions in my CNN model which is fairly deep model. Below shown is the snapshot of the model at epoch 71/72.

Epoch 00071: val_loss improved from 0.48918 to 0.48610, saving model to model.h5
625/625 [==============================] - 55s 87ms/step - loss: 0.4496 - dice_coef: 0.8399 - iou: 0.7251 - recall: 0.8658 - precision: 0.9139 - val_loss: 0.4861 - val_dice_coef: 0.8491 - val_iou: 0.7408 - val_recall: 0.8650 - val_precision: 0.9055 - lr: 1.0000e-04
Epoch 72/125
625/625 [==============================] - ETA: 0s - loss: 0.4436 - dice_coef: 0.8423 - iou: 0.7286 - recall: 0.8676 - precision: 0.9152
Epoch 00072: val_loss did not improve from 0.48610
625/625 [==============================] - 53s 85ms/step - loss: 0.4436 - dice_coef: 0.8423 - iou: 0.7286 - recall: 0.8676 - precision: 0.9152 - val_loss: 0.4915 - val_dice_coef: 0.8470 - val_iou: 0.7376 - val_recall: 0.8628 - val_precision: 0.9049 - lr: 1.0000e-04

Jaccard loss function - it is the division of intersection of false positive and false negative over the union.
CIFAR-10 is itself really big data and the pictures used in it are even hard for a human to categorize the image.

Dice loss function - it is the division of intersection over the intersection with the addition of union.
Here we notice that the loss for the dice is less than Jaccard. because the dice function focuses more on the true positive and tries to decrease the error.


# Q2. Dataset Description

## The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

The label classes in the dataset are:

0- airplane 
1- automobile 
2- bird 
3- cat 
4- deer 
5- dog 
6- frog 
7- horse 
8- ship 
9- truck

# Q3. Details

First, I loaded the CIFAR-10 dataset
Then I normalized the data between the value 0 to 1 making the data fit for the CNN model.
Then I made the DeepCNN model. The model consists of 6 Conv2D layers.
Batch normalization after each layer. after I included batch normalization I found out that the results were changed and were much better. It happens because when the data is passed after each layer, the data is been manipulated and the 'elu' activation function after each layer triggered a neuron to predict the data bt as we go deeper and deeper the triggered neuron dominated over other resulting in giving the wrong prediction.
But after I added the batch normalization, the neurons were normalized after each layer in giving the fair data distribution after each layer. Hence the accuracy of the model increased.
As the model was training I found that the weights after remained intact meaning when the data was passed from the first Conv2D layer it added some weights and the data was then passed to the next layer. but in some cases, the weights were increasing drastically resulting in a very high error rate. After that, I did my research and come across the concept of weight decay.
Weight decay controls the weight from increasing too high. 
Then I compile the model with 'dice_coef, iou, Recall(), Precision()' metrics and  RMSprop optimizer with learning rate of 0.0001 and I added the decay rate as well of 1e-6
Then train the model over 125 epochs and use the callback function of TensorFlow for visualization of how the model is working.
I also used the data augmentation by Keras on the data to train the model on a variety of augmented data.
In augmentation, the data was flipped horizontally, rotated with an angle of 15 degrees and changed the height and width of the data.
Also saving the final model  

Final result is 

Epoch 125/125
625/625 [==============================] - ETA: 0s - loss: 0.3760 - dice_coef: 0.8614 - iou: 0.7576 - recall: 0.8857 - precision: 0.9257
Epoch 00125: val_loss did not improve from 0.44047
625/625 [==============================] - 56s 90ms/step - loss: 0.3760 - dice_coef: 0.8614 - iou: 0.7576 - recall: 0.8857 - precision: 0.9257 - val_loss: 0.4497 - val_dice_coef: 0.8580 - val_iou: 0.7544 - val_recall: 0.8741 - val_precision: 0.9088 - lr: 1.0000e-05

Reference
https://appliedmachinelearning.blog/2018/03/24/achieving-90-accuracy-in-object-recognition-task-on-cifar-10-dataset-with-keras-convolutional-neural-networks/
