This folder contains the .prototxt files defining LeNet-5. Training 
can be done with the following command (assuming caffe is already 
installed)

./caffe train -solver lenet_solver.prototxt

Testing can be done with the following command

caffe test -model lenet_test.prototxt -weights test_caffe_X.caffemodel
