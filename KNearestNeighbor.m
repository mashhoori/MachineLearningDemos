
clear 
close all
addpath('.\MNIST\');

%% Loading the Data
trainImages = loadMNISTImages('\MNIST\train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('\MNIST\train-labels.idx1-ubyte');
testImages = loadMNISTImages('\MNIST\t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('\MNIST\t10k-labels.idx1-ubyte');

%% Reducing the size of the training set (to make the training process faster)
ratio = 0.2; % The ratio of the samples that we want to use for training 
numTrain = floor(ratio * numel(trainLabels));
trainIndices = randsample(numel(trainLabels), numTrain, false);

trainImages = trainImages(:, trainIndices);
trainLabels = trainLabels(trainIndices);

numTrain = numel(trainLabels);
numTest  = numel(testLabels) ;

%% Predicting label for each test sample using the nearst neighbor classifier
k = 5; % number of neighbors
labelPredicted = zeros(size(testLabels));
for i = 1:numTest
    i    
    df = bsxfun(@minus, trainImages, testImages(:, i));  
    dist = sum(df .^ 2);
    [~, indices] = sort(dist);  % Not the best way to find the k smallest number, but the easiest way!
    labelPredicted(i) = mode(trainLabels(indices(1:k)));    
end

%% Computing the accuracy of predictions
accuracy = sum( labelPredicted == testLabels) / numTest;
fprintf('The prediction accuracy is: %0.2f percent \n', accuracy * 100);
