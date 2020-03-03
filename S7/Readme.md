Team : Hemendra Srinivasan
	   Umesh Singh(mshsingh772)

- The model achieves an accuracy of 81% with 35 epochs under the given contraints of parameters less than 1M.

- The model includes dilation in convblock2 and Depthwise separable convolution in convblock3. The test data normalisation includes RandomHorizontalFlip and RandomCrop

- The modularised code includes files:
	
    a. cifar10.py - 
		includes model architecture
		
    ![Image description](https://github.com/mshsingh772/eva/blob/master/S7/imgs/Cifar10_Arch.PNG)

	c. imagetransforms.py-
		provides the transformation for test and train data
	
	b. dataloader.py - 
		loads the train and test data with their correponding transformations
	
	d. train_test_model.py - 
		performs check for the available devices and provides function for running the model
