
###### Team members:
    - Umesh Singh (mshsingn772@gmail.com)
    - Hemendra Srinivasan (hemendra1111@gmail.com)


- The model achieves an accuracy of 85.67% with 24 epochs.

- The model makes use of ResNet18 architecture for training cifar10

- The modularised code includes files:
	
    a. the model folder contains different model architectures for cifar10 like,

		- custom architecture 
		- ResNet18 architecture
		- custumResNet architecture

	c. imagetransforms.py -

		provides the pytorch transformation for test and train data and albumentation transformation for train as well
	
	b. dataloader.py - 

		loads the train and test data with their correponding transformations
	
	d. train_test_model.py - 

		performs check for the available devices and provides function for running the model

	e. utils.py -

		contines the utility functions like display of model summary, calculation of mean and standard deviation

	f. gradcam.py

		contains the necessary methods to implement gradcam
	
	g. plot_grad.py

		contains the code for plotting the missclassified classes and applying gradcam on them
	
	h. lr_finder.py

		contains the lr finder implementaions 