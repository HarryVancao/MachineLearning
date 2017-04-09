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


%% 

% use cov function to find covariance

% 
% 
%     % extract dimensions of data
%     data_dim = size(input_data, 1);
%     target_dim = min(target_dimensions, data_dim);
%     
%     % extract number of datapoints.
%     num_samples = size (input_data, 2); 
%     
%     % center data to include zero. 
%     centered_data = input_data; 
%     for i = 1:data_dim
%         centered_data(i, :) = centered_data(i, :) - mean(centered_data(i, :)) ; 
%     end 
%     %reduced_training = centered_data; 
%     
%     covariance = cov(centered_training(:)) 
%     
%     % initialize transformation matrix. 
%     V = eye (data_dim, target_dim);
%     V = V ./ norm(V(:, 1));
%     
%     [V, eigen_value] = eigs (centered_data * centered_data', target_dim); 
%     %V = eigen_value * V
%     
%     % find 1st principal component: the eigenvector corresponding to the
%     % maximal eigenvalue of X'*X. 
% %     [eigenvector, eigenvalue] = eigs(centered_data * centered_data', 1);
% %     V(:, 1) = eigenvector ;
% %     for i = 2:target_dimensions 
% %         % subtract off the current largest principal component
% %         centered_data = centered_data - ( (V(:, i - 1) * V(:, i - 1)') * centered_data);
% %         % find next largest principal component.
% %         [eigenvector, eigenvalue] = eigs(centered_data * centered_data', 1);
% %         V(:, i) = eigenvector;
% %     end    
%     
%     % apply the transformation to both training AND test data.
%     reduced_training = V' * reduced_training;
%     reduced_test = V' * test_data; 