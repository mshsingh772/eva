###### Team members:
	- Hemendra Srinivasan (hemendra1111@gmail.com)
    - Umesh Singh (mshsingh772@gmail.com)
    


- The model achieves an accuracy of 86% with 15 epochs.

- The model makes use of ResNet18 architecture for training cifar10

- The modularised code includes files:
	
    a. the model folder contains different model architectures for cifar10 like,

		- custom architecture 
		- ResNet18 architecture

	c. imagetransforms.py -

		provides the transformation for test and train data
	
	b. dataloader.py - 

		loads the train and test data with their correponding transformations
	
	d. train_test_model.py - 

		performs check for the available devices and provides function for running the model

	e. utils.py -
	
		contains the utility functions like display of model summary 