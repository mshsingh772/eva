
###### Team members:
    - Umesh Singh (mshsingn772@gmail.com)
    - Hemendra Srinivasan (hemendra1111@gmail.com)


- The model achieves an test accuracy of 55.15% with 50 epochs.

- The model makes use of ResNet18 architecture for training TinyImagenet_200

- The modularised code includes files:
	
    a. the model folder contains different model architectures,

		- customeResNet
		- mnistmodel
		- cifar10 architecture 
		- ResNet18 architecture

	c. imagetransforms.py -

		provides the pytorch transformation for test and train data and albumentation transformation for train as well.
		Has been modified to create transforms by considering the dataset directly to albumentations
	
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
	
	i. Tiny_Imagenet_200.py

		contains the code for dowloading the tiny imagenet dataset and also snippets for train test split as well