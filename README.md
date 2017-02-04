# Disclaimer
The following training procedures are based on the rules of the contest. Since there were no time restrictions in the training and testing time, I allow myself to use smaller learning rates, larger training/validation sets, bigger models, etc. 
In order to be able to speed-up the evaluation of the models, I also provide you with some parameter configuration hints. **This, however, may lead to worse performance in accuracy on the test set. I, therefore, advise you to stick to the default settings, as long as you have the computational resources for that.**

# Requirements
The code is written in Python2.7 using the [Anaconda](https://www.continuum.io/downloads) environment. However, only the external packages *numpy*, *scipy* and *matplotlib* are imported so it will not be an issue to skip the full Anaconda installation.

On the hardware side, you will need at least 3GB of RAM for the simpler models and preferably 10GB for the GPs.  

# Project structure and data setup
In the root there are the folders:
* data
* report

and the packages:
* experiments
* knn
* logreg
* nn
* gp
* numerics
* utils

Please, put the raw data files, i.e. 
- t10k-images-idx3-ubyte
- t10k-labels-idx1-ubyte
- train-images-idx3-ubyte
- train-labels-idx1-ubyte 

in the *data* folder as they are expected to be found there.

# Training and testing

Every model implements a `fit(train_data, **kwargs)` and `predict(test_data, **kwargs)` methods which allow for training and testing respectively. In order to encapsulate the details about the data parsing, formatting and so on, you can find in the main.py module the method `evaluate_[modelname]([train_from_scratch=True, verbose=True])` which runs both! training (validation if necessary) and testing. If you want to switch off training (which is, unfortunately, not available for all  models as it would require the storage (and submission) of binary (very) large objects, like for example the covariance matrices in GPs!)

# Models
### k-NN
There is nothing much to be tuned for the training of this model as it is automatically cross-validated (5-folds over the range [1,2,...,10] for the neighbourhood size *k*).

**Speed-up hints:** A 3-fold cross validation is also good enough to find the best value for *k* so if you need to, you can edit the hardcoded `validator = KFoldCrossValidation(data=data_train, k=5)` in the *experiments/evaluate_models.py* and then in `evaluate_knn(...)`.

### LogReg
Same as above: it is also cross-validated (5-folds over the range [0., 1e-5, 1e-4, 1e-3, 1e-2] for the parameter governing the regularisation strength *reg*).

**Speed-up hints:** Again 3-fold cross validation might be well behaving too, although there is not much of a need to touch the default parameters as it works already fast enough (7m), at least in comparison to the other models :)

### ConvNet
This is the model which requires probably least amount of memory but trains the longest, so I advise you to leave `evaluate_convnet(train_from_scratch=False)` as shown in the main. The training procedure I conducted is as follows:
I trained with the default solver parameters and most notably the learning rate which was set to 0.005 for about 4,5 epochs. The validation accuracy after the second epoch was about 98.5% and after the 4th -- around 99%.
Then I trained for 4 more epochs with learning rate 0.001 and managed to achieve a validation accuracy of about 99.25%. I didn't try to lower the learning rate any further and continue from there, as I didn't have the patience with those navie forward and backward convolution layers. To give you a sense of the time needed: 4 epochs are completed after about 12-14 hours of training.

**Speed-up hints:** I implemented the convolutions both with for-loops and using the scipy.signal.convolve method. The latter is about 3-4 times faster and should be preferred unless there are some restrictions on that. It is also set as the default option when the constructor of the ConvolutionalNeuralNetwork is called in the corresponding `evalutate_convnet` method.

### BasicNet
For comparison, I also tried a simpler one-layer neural network with 500 units. It saturates at about 97.7-98% validation accuracy after around 30 epochs (which take approx. 5 minutes to complete). The test accuracy is also in that range.

### CGPs
The classification Gaussian processes can be run both in "multiclass" and "one-vs-rest" modes. The former trains a single GP with extended covariance matrix and latent function for each class. The latter trains 10 binary classifiers which then combined produci a multi-class predictions. They also achieve much better accuracy, probably due to a bug in the somewhat involved implementation of the multiclass GP. 
The data sets for each of the binary classifiers is a stratified subset from the whole training set, where 50% of the samples belong to one single class and 50% to the rest. 
The size of the training set for both GP classifiers can be adjusted and in the experiments for the report I used 2000 for the multiclass one and 6000 for each binary. These numbers are set in the `evaluate_gp` methods. 

**Speed-up hints:** A way to speed up considerably the evaluation is to decrease the size of the training data for each class. Since the improvement is relatively small (3-4%) between runs with with data sizes of 500, 1000 and 2000 samples per class (for the binary case) you can consider decreasing the `train_data_limit` parameter in the `evaluate_gp` function in the experiments/evaluate_models.py.



















