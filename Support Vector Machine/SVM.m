%% PARAMETERS
clc; 
%clear;
use_pca = false;
use_sheared = false; 
kernel_type = 4; 
target_dimensions = 50; 
c = 10;
num_test_examples = 10000; 

% load mnist data
% functions downloaded from: 
%http://ufldl.stanford.edu/wiki/resources/mnistHelper.zip
images = loadMNISTImages('train-images-idx3-ubyte');
labels = loadMNISTLabels('train-labels-idx1-ubyte');
test_images = loadMNISTImages('t10k-images-idx3-ubyte');
test_label = loadMNISTLabels('t10k-labels-idx1-ubyte'); 

% use shear test set? 
if use_sheared == true 
    test_images = Shear(test_images, 0.5); 
end 

% Dimensionality Reduction
if use_pca == true 
    % performa PCA on the data. 
    components = pca(images');
    train = components(:, 1:target_dimensions)' * images;
    test = components(:, 1:target_dimensions)' * test_images(:, 1:num_test_examples); 
else 
    % performa LDA on the data.
    %Mdl = fitcdiscr(images', labels);
    labels = labels + ones(60000,1);
    [train, test] = LDA(images, labels, test_images, c, target_dimensions);
    labels = labels - ones(60000, 1);
end 

% training of SVM (select kernel)
if kernel_type == 0 
    % 84.65% on 10 dimensions
    the_model = svmtrain(double(labels), train', '-b 1 -q -t 0');
elseif kernel_type == 1 
    % 93.85% on 10 dimensions
    the_model = svmtrain(double(labels), train', '-b 1 -q -t 1');
elseif kernel_type == 2
    % 80.46% on 50 dimensions
    the_model = svmtrain(double(labels), train', '-b 1 -q -t 2');
else 
    the_model = svmtrain(double(labels), train', '-b 1 -q');
end 

% test SVM model
% 94% with PCA to 10 dims.
% 93.13% with PCA to 9 dims.
% 90.29% with LDA to 9 dims.
[l, ~, p] = svmpredict(double(test_label), test', the_model, '-b 1 -q');
sum(l == test_label) / num_test_examples
