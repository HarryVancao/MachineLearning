clear;
clc;
% Load in the datasets!
face = load('../Data/data.mat');
face = face.face;
illumination = load('../Data/illumination.mat');
illumination = illumination.illum; 
pose = load ('../Data/pose.mat');
pose = pose.pose; 

%% Reform face data 

% number of samples to train per class
total_per_class = 3; 
train_per_class = 2; 
test_per_class = total_per_class - train_per_class; 

num_samples = 600; 
c = 200;

face = reshape(face, 21*24, 600);
face_train = zeros (21*24, train_per_class*c); 
face_test = zeros(21*24, test_per_class * c);
train_label = zeros(train_per_class*c, 1); 
test_label = zeros(test_per_class*c, 1); 

for class = 1:c
    for i = 1:train_per_class
        face_train(:, train_per_class*(class-1) + i) = face(:, class*total_per_class - 3 + i);
        train_label(train_per_class*(class-1) + i) = class;
    end 
    for i = 1:test_per_class
        face_test(:, test_per_class*(class-1) + i) = face(:, class*total_per_class - 3 + i + train_per_class); 
        test_label(test_per_class*(class-1) + i) = class; 
    end 
end 
%[face_train, face_test] = PCA(face_train, face_test, 200); 
%[face_train, face_test] = LDA (face_train, train_label, face_test, c); 
%guess = Bayes_Classify (face_train, train_label, face_test, c, 1);
guess = Nearest_Neighbor_Classify(face_train, train_label, face_test, 2);
accuracy = sum(guess == test_label) / size (test_label, 1)

%% Reform Pose Data 
% number of samples to train per class
total_per_class = 13; 
train_per_class = 7; 
test_per_class = total_per_class - train_per_class; 
c = 68;

pose = reshape(pose, 48*40, 13*68);
pose_train = zeros (48*40, train_per_class*c); 
pose_test = zeros(48*40, test_per_class * c);
train_label = zeros(train_per_class*c, 1); 
test_label = zeros(test_per_class*c, 1); 

for class = 1:c
    for i = 1:train_per_class
        pose_train(:, train_per_class*(class-1) + i) = pose(:, (class - 1)*total_per_class + i);
        train_label(train_per_class*(class-1) + i) = class;
    end 
    for i = 1:test_per_class
        pose_test(:, test_per_class*(class-1) + i) = pose(:, (class - 1)*total_per_class + i + train_per_class); 
        test_label(test_per_class*(class-1) + i) = class; 
    end 
end 

%[pose_train, pose_test] = LDA (pose_train, train_label, pose_test, c);
%[pose_train, pose_test] = PCA(pose_train, pose_test, 200);
guess = Nearest_Neighbor_Classify(pose_train, train_label, pose_test,  2)
%guess = Bayes_Classify (pose_train, train_label, pose_test, c, 1);
accuracy = sum(guess == test_label) / size (test_label, 1)


%% Reform Illumination
% number of samples to train per class
total_per_class = 21; 
train_per_class = 17; 
test_per_class = total_per_class - train_per_class; 
c = 68;

illumination = reshape(illumination, 1920, 21*68);
illumination_train = zeros (1920, train_per_class*c); 
illumination_test = zeros(1920, test_per_class * c);
train_label = zeros(train_per_class*c, 1); 
test_label = zeros(test_per_class*c, 1); 

for class = 1:c
    for i = 1:train_per_class
        illumination_train(:, train_per_class*(class-1) + i) = illumination(:, (class - 1)*total_per_class + i);
        train_label(train_per_class*(class-1) + i) = class;
    end 
    for i = 1:test_per_class
        illumination_test(:, test_per_class*(class-1) + i) = illumination(:, (class - 1)*total_per_class + i + train_per_class); 
        test_label(test_per_class*(class-1) + i) = class; 
    end 
end
%[illumination_train, illumination_test] = PCA(illumination_train, illumination_test, 200);
%[illumination_train, illumination_test] = LDA(illumination_train, train_label ,illumination_test, c);
guess = Nearest_Neighbor_Classify(illumination_train, train_label, illumination_test, 2); 
%guess = Bayes_Classify (illumination_train, train_label, illumination_test, c, 1);
accuracy = sum(guess == test_label) / size (test_label, 1)
