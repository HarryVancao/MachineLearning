%% Fisher's Linear Discriminant Analysis
function [ reduced_training, reduced_test ] = LDA( train_data, train_label, test_data, c)
    % This function transforms the data c - 1 dimensional data where c is
    % the number of classes that we are discriminating. This is based on
    % Fisher's Linear Discriminant analysis which attempts to maximally
    % divide the data into c reigons through a transformation. The function
    % applies the transformation to both the training and the test data. 
    
    num_samples = size(train_data, 2); 
    num_dims = size(train_data, 1); 
    
    % compute mean for each class
    label_count = zeros(c, 1);
    mu = zeros(num_dims, c); 
    for i = 1:num_samples 
        mu(:, train_label(i)) = mu(:, train_label(i)) + train_data(:,i);
        label_count(train_label(i)) = label_count(train_label(i)) + 1 ;
    end 
    mu = mu ./ label_count'; 
    
    Within_Scatter = zeros(num_dims, num_dims); 
    % compute the within scatter matrices.
    for i = 1:num_samples
        Within_Scatter = Within_Scatter + (train_data(:, i) - mu(:, train_label(i))) * (train_data(:, i) - mu(:, train_label(i)))'; 
    end 
    
    % Total mean M
    M = zeros(num_dims, 1); 
    for i = 1:c
        M = M + label_count(i) * mu(:, i);
    end 
    M = M / num_samples;
    
    % Compute the Between scatter matrix
    Between_Scatter = zeros(num_dims);
    for i = 1:c
        Between_Scatter = Between_Scatter + (label_count(i) * ((mu(i) - M) * (mu(i) - M)')); 
    end
    
    %a = [det(Between_Scatter) det(Within_Scatter)]
    
    % solve for optimal W and yield the reduced dimension training
    % set. (correct within/between scatter from being singular)
    % W = zeros (num_dims, c - 1); 
    
    Between_Scatter = Between_Scatter + 1 * eye(num_dims);
    Within_Scatter = Within_Scatter + 1 * eye(num_dims);
    
    [W,EV] = eigs(Between_Scatter, pinv(Within_Scatter), c-1);
    
    % W is the optimally seperating transform
    reduced_training = W' * train_data; 
    reduced_test = W' * test_data;    
end
