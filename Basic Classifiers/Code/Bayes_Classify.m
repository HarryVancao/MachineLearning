%% Bayes Rule Classifier
function [guess_vector] = Bayes_Classify (train_data, train_label, test_data, c, lambda)
    % this function takes in training data classifies the passed in test data
    % based on bayes rule. Please read the following for details on how to
    % format/pass in data to the function. 

    % train_data/label := the training dataset/label
    % test_data := the test dataset
    % c := number of all possible labels 

    % Constraint: we assume that train_data is a matrix with each column
    % representing a sample with each row representing a feature.
    % images for example, must be reformed to a single feature vector.

    % guess_vector := a vector with the assigned labels to each test data
    % entry.

    % calculate dimensions of data
    num_dims = size(train_data,1); % # dim of feature vects
    num_samples = size(train_data, 2); % # training samples
    num_test_samples = size(test_data, 2); % # test data samples 
    
    % counts how many samples of each label we have encountered.
    label_count = zeros(c, 1);
        
    % compute the MLE estimate of mean vector for each class. 
    mu = zeros(num_dims, c);
    for i = 1:num_samples
        mu(:, train_label(i)) = mu(:, train_label(i)) + train_data(:,i);
        label_count(train_label(i)) = label_count(train_label(i)) + 1;
    end 
    mu = mu ./ label_count'; 
    
    % compute MLE estimate of covariance matrix
    sigma = zeros (num_dims, num_dims, c); 
    for i = 1:num_samples 
       % variance(:, train_label(i)) = variance(:, train_label(i)) + (train_data(:,i) - mu(:, train_label(i))).^2;
       sigma(:, :, train_label(i)) = sigma(:, :, train_label(i)) + (train_data(:,i) - mu(:, train_label(i)))' * (train_data(:,i) - mu(:, train_label(i)) );
    end 
    
    % reused from computation of mu
    for i = 1:c 
        sigma(:, :, i) = sigma(:, :, i) / label_count(i); 
        %if det(sigma(:, :, i)) == 0 
            sigma(:, :, i) = sigma(:, :, i) + lambda * eye(num_dims);
        %end
    end 
    
    % compute discriminant functons from log-likelihood function
    
    % Variables holding discriminant function parameters/mle estimates 
    %sigma = zeros(num_dims, num_dims, c); 
    sigma_inv = zeros(num_dims, num_dims, c);
    w2 = zeros (num_dims, num_dims, c);
    w1 = zeros (num_dims, c);
    w0 = zeros (c, 1);
    
    % for each class, compute sigma AND discriminant function paramters. 
    for i = 1:c        
        %add correction factor to prevent covariance from being singular%
        if (det(sigma(:, :, i)) == 0) 
            sigma(:, :, i) = sigma(:, :, i) + (lambda) * eye(num_dims, num_dims);
        end 
        % inverstion of covariance matrix
        %if (det(sigma(:, :, i)) == 0)
            %sigma(:, :, i) = pinv(sigma(:, :, i)); 
        %else 
        sigma_inv(:, :, i) = inv(sigma(:, :, i));
        %end 
        
        % square parameter 
        w2(:,:,i) = (-1/2) * sigma_inv(:, :, i); 
        % linear parameter
        w1(:,i) = mu(:, i)' * sigma_inv(:, :, i);
        % constant parameter
        w0(i) = (-1/2)*( (mu(:, i)' * sigma_inv(:,:,i) * mu(:, i))  + log(det(sigma(:, :, i))) );        
    end
    
    % Here, our model is complete and we have generated our discriminant
    % functions. Now, we find the maximum output value and classify on that
    
    disp('Done training... Now deciding.')

    guess_vector = zeros(num_test_samples, 1);
    for i = 1:num_test_samples
        max_index = 1;
        max_g = test_data(:, i)' * w2(:, :, 1) * test_data(:, i) + w1(:, 1)' * test_data(:, i);
        max_g = max_g + w0(1);
        for j = 1:c
            g = (test_data(:, i)' * w2(:,:,j) * test_data(:, i)) + (w1(:,j)' * test_data(:, i));
            g = g + w0(j);
            if (g >= max_g)
                max_g = g;
                max_index = j;
            end 
        end
        guess_vector(i) = max_index; 
    end
end 