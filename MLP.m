% Implementation of a multilayer neural network with one hidden layer 
% Code written by: Ali Mashhoori August 2016

% 0.972 


function MLP()

addpath('.\MNIST\');

%% Loading the Data

trainImages = loadMNISTImages('\MNIST\train-images.idx3-ubyte');
trainLabels = loadMNISTLabels('\MNIST\train-labels.idx1-ubyte');
testImages = loadMNISTImages('\MNIST\t10k-images.idx3-ubyte');
testLabels = loadMNISTLabels('\MNIST\t10k-labels.idx1-ubyte');

[trainImages, mu, sigma] = zscore(trainImages, [], 2);
testImages =  bsxfun(@minus, testImages, mu);
testImages =  bsxfun(@rdivide, testImages, (sigma+eps));

trainLabels = dummyvar(trainLabels+1);

%%
targetDim = 10;
inputDim = 28*28;
nHidden = 100;

params = DefineParams(inputDim, nHidden, targetDim);

%%
params = Train(trainImages', trainLabels, params);
res = Predict(testImages', params);

accuracy = sum(testLabels == res) / numel(res);
fprintf('The accuracy is: %0.3f \n', accuracy);

end

function params = DefineParams(inputDim, nHidden, nOutput)
 
scale = sqrt(6 / (inputDim + nOutput)); 
wInpToHid = rand(inputDim + 1, nHidden)* scale - scale/2;

scale = sqrt(6 / nHidden); 
wHidToOut = rand(nHidden + 1, nOutput) * scale - scale/2;

params.wInpToHid = wInpToHid;
params.wHidToOut = wHidToOut;

end
 

function [output] = ForwardPass(batch, params)

batch = [ones(size(batch, 1), 1),  batch];
hidIn = batch * params.wInpToHid;
hidOut = tanh(hidIn);

hidOut = [ones(size(hidOut, 1), 1),  hidOut];
outIn = hidOut * params.wHidToOut;

maxValues = max(outIn, [], 2);
netInput = bsxfun(@minus, outIn, maxValues);
netInput = exp(netInput);
probs = bsxfun(@rdivide, netInput, sum(netInput, 2)); 

output.input = batch;
output.hidden = hidOut;
output.probs = probs;

end

function [cost, grad] = ComputeCostAndLastLayerGradient(netOutput, target)

probs = netOutput.probs;
probs(probs == 0) = 1e-20;
cost = sum(sum(-1 * target .* log(probs)));

grad = probs - target;

end

function gradInfo = BackPropagate(grad, net, params)

batchSize = size(grad, 1);
hiddenGrad = (net.hidden' * grad / batchSize);
 
tmp = grad * params.wHidToOut(2:end, :)';
res = (1 - net.hidden(:, 2:end).^2); 
tmp2 = tmp .* res;

inputGrad = (net.input' * tmp2 / batchSize);

gradInfo.inputGrad = inputGrad;
gradInfo.hiddenGrad = hiddenGrad;

end

function params = UpdateParams(gradInfo, params, learningRate)

params.wInpToHid = params.wInpToHid - learningRate * gradInfo.inputGrad;
params.wHidToOut = params.wHidToOut - learningRate * gradInfo.hiddenGrad;

end

function params = Train(input, target, params)

batchSize = 100;
numBatch = size(input, 1)/batchSize;

batches = cell(1, numBatch);
for i=1:numBatch
    batches{i}.input = input((i-1)*batchSize+1:i*batchSize, :);
    batches{i}.target = target((i-1)*batchSize+1:i*batchSize, :);
end


lr = 0.05;
for iter = 1:50
    costIter = 0;
    for b = 1:numBatch        
    
        output = ForwardPass(batches{b}.input, params);
        [cost, grad] = ComputeCostAndLastLayerGradient(output, batches{b}.target);
        costIter = costIter + cost;
        
        gradInfo = BackPropagate(grad, output, params);
        params = UpdateParams(gradInfo, params, lr);            
        
    end
    
    if(iter < 10)
            lr  = lr * 0.999;
    end 
    fprintf('The cost after %d iterations is %f \n', iter, costIter / size(input,  1));

end

end

function labels = Predict(input, params)

    output = ForwardPass(input, params);
    [~, labels] = max(output.probs, [], 2);
    labels = labels - 1;
    
end



