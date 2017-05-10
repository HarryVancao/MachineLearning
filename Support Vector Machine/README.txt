This is my implementation of SVM using LibSVM to classify the 
digits from the MNIST Dataset. 

Here is a breakdown on each file: 

SVM.m - This is where the SVM is implemented. It is the only
	file that is needed to run my code. Everything else 
	just supports this script.

Shear.m - this is a function file that shears a dataset and
	returns a new test set.

LDA.m - This is a function that reduces the dimensionality of
	the input data using the LDA algorithm.

create_db - This is a simple script that writes image data 
	to a idx-ubyte format. This is useful for creating 
	new test datsets to run in caffe.

libsvmread, libsvmwrite, loadMNISTX, svmtrain, svmpredict 
	These are all files containing the LibSVM library.

Everything else is either a training/testing datset.

To run the script, open SVM.m and adjust the parameters at the
beginning of the file and run. The function prints out 
the performance of the classifier on the test dataset and 
the guessed label vector is in a variable named l. 


