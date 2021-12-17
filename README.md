# Vision-based-Vehicle-Detection-and-Tracking

For Object Classification, we implemented a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training dataset of images. In this way, we can highlight the important features of the object so that the input data size can be largely reduced. 

Before training, we normalized the features to have zero mean and unit variance using sklearn’s StandardScaler. By using sklearn's StandardScaler, we can guarantee that each feature contributes equally to the training loss and therefore make the model converge faster.

Since there are only two types of objects (Vehicle/ Non-vehicle), linear SVM is quite suitable for this project. Now we are using sklearn to train the data.

For object detection, we utilized the sliding window technique to search vehicles on given images. We also implemented pyramid representation to search vehicles on different scales. In this way, we can get better performance.

train.py is for sklearn svm model training and saving.

detector.py is running model on test data and save the result as a gif.
