This is a Matlab implementation of four machine learning algorithms.

The code is split into four functions. The intention is that the functions enforce a format of data and classify on that. Thereby 
decoupling the data format from the actual classification. As a consequence, it is up to the caller of the functions to modify and
adjust their dataset to adhere with the format. More documentation on this format, which is fairly simple, can be found in the 
function files. 

The functions that implement classifiers are:

Bayes_Classifier (train_data, train_label, test_data, c)
and 
Nearest_Neighbor_Classifier (train_data, train_label, test_data, k) 

These output a "guess" vector which holds the predicted labels of each entry in test_data. More documentation of their functionality 
can be found in the .m files with the associated names. 


On the other hand, the two remaining functions: 

PCA (train_data, test_data, target_dimension) 
and
LDA (train_data, train_label, test_data, c) 

implement dimensionality reduction algorithms which they are named after. They output a reduced dimensional version of train/test data 
which are immediately ready to be passed into Bayes_Classifier/Nearest_Neighbor_Classifier functions. Again, more documentation of their
functionality can be found in the associated .m files. 


The code of these functions can be readily accessed in the current folder. Supporting these functions, a single main.m script demonstrates the
usage of the funcitons. 

To run the script file, navigate to the current directory in matlab and execute the top section first (titled: Load Datasets) 
and then each subsequent section showcases a different combination of using the four functions on various datasets that were given 
in the /Data folder. 
