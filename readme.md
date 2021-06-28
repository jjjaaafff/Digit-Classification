# Digit Classification


* Environment
  
  * Python 3.7.0
  * numpy 1.18.1
  * torch 1.6.0
  * torchvision 0.7.0

* Dataset

    The Street View House Numbers (SVHN) Dataset with Format 2: Cropped Digits
    The link for SVHN dataset is [http://ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/)

* Model

  * Logistic regression (+ Ridge/LASSO loss)
  * Linear Discriminant Analysis (LDA）
  * Kernel-based logistic regression + LASSO loss
  * SVM (Linear and non-linear with various kernels)
  * Convolutional Neural Networks

* Project Structure

  * dataset: store the *mat* format dataset and hog_extracted *npy* data
  * model: store the trained model
  * **data_preprocess.py**: extract the histograms of oriented gradients
(HOG) feature of pictures in original dataset and store them.
  * **others**: implementation of different classification models 

* Details can be found in report
