%% Principal Component Analysis 
function [ reduced_training, reduced_test ] = PCA( input_data, test_data, target_dimensions )
    % This function takes in the input data and reduces the dimensionality
    % of the data to the target dimensionality. This is achieved through
    % the PCA algorithm. 

    % extract dimensions of data
    data_dim = size(input_data, 1);
    target_dim = min(target_dimensions, data_dim);
    % extract number of datapoints.
    
    % center data. 
    centered_data = input_data; 
    for i = 1:data_dim
        centered_data(i, :) = centered_data(i, :) - mean(centered_data(i, :)) ; 
    end 
    
    % get the covariance of the input data
    covariance = cov(centered_data');
    
    % find the vectors that induce the largest variance.
    [V, eigen_value] = eigs (covariance, target_dim); 
    reduced_training = V' * input_data;
    reduced_test = V' * test_data; 
end
