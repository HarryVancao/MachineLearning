%% K-Nearest-Neighbor Classifier
function [guess_vector] = Nearest_Neighbor_Classify (train_data, train_label, test_data, k) 
    % This function takes training data and classifies test data based on the k
    % nearest neighbor rule. Please read the following to see how data should
    % be formatted and passed into the classifier. 

    % train_data/label := the training dataset/label
    % test_data := the test dataset
    % k := the number of neighbors that are allowed to vote. 

    % Constraint: we assume that train_data is a matrix with each column
    % representing a sample. each row represents a variable (mutually independent)
    % images for example, must be reformed to a single feature vector. 

    % guess_vector := a vector with the assigned labels to each test data
    % entry.
    
    % calculate number of test/training samples
    num_test_samples = size(test_data, 2);
    num_train_samples = size(train_data, 2);
    
    % output vector holding the guessed label of the ith test sample
    guess_vector = zeros(num_test_samples, 1); 
    
    for i = 1:num_test_samples
        % create datastructure 
        k_nearest = zeros(k, 1);
        k_nearest_label = zeros(k, 1); 
        
        %init data-structure
        k_nearest(:) = Inf; 
        k_nearest_label(:) = Inf; 
        
        % compute distance from first training sample and store label. 
        k_nearest(1) = norm(test_data(:, i) - train_data(:, 1));
        k_nearest_label(1) = train_label(1);
        
        % find the k-nearest neighbors. 
        for j = 2:num_train_samples
            cur_norm = norm(test_data(:, i) - train_data(:, j)); 
            [valmax, argmax] = max(k_nearest); 
            if (cur_norm < valmax) 
                k_nearest(argmax) = cur_norm; 
                k_nearest_label(argmax) = train_label(j); 
            end 
        end
        % k-nearest neighbors vote. Ignores infinities. 
        guess_vector(i) = mode(k_nearest_label( : )); 
    end
end 