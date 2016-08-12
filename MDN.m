% Implementation of a mixture density network with one hidden layer and
% spherical covariance matrix for each feature
% Code written by: Ali Mashhoori August 2016

function MDN()

% Exmaple taken from: Bishop, Christopher M. "Mixture density networks." (1994).

numCenters = 3;
targetDim = 1;
inputDim = 1;
nHidden = 20;

params = DefineParams(inputDim, nHidden, targetDim, numCenters);


f = @(t) t + (rand(size(t)) * 0.2) - 0.1 + 0.3*sin(2*pi*t);
t = linspace(0,1, 1000);
t = t(:);
y = f(t);
y = zscore(y);

plot(y, t, '.'); 
hold on

params = Train(y, t, params);
res = GetMeanAndSTD(y, params);

plot(y, res.predMean, '.k')
hold on 
plot(y, res.predMean + res.predSigma, '.g')
plot(y, res.predMean - res.predSigma, '.g')
plot(y, res.predMaxComp, '.r')

end

function params = DefineParams(inputDim, nHidden, targetDim, numCenters)
 
nOutput = numCenters + (targetDim * numCenters) + numCenters;

scale = sqrt(6 / (inputDim + nOutput)); 
wInpToHid = rand(inputDim + 1, nHidden)* scale - scale/2;

scale = sqrt(6 / nHidden); 
wHidToOut = rand(nHidden + 1, nOutput) * scale - scale/2;

params.wInpToHid = wInpToHid;
params.wHidToOut = wHidToOut;
params.numCenters = numCenters;
params.centerProbIndices = 1:numCenters;
params.centerLocIndices  = numCenters+1: numCenters + (targetDim * numCenters);
params.centerVarIndices = params.centerLocIndices(end)+1:nOutput;

end


function [output] = ForwardPass(batch, params)

batch = [ones(size(batch, 1), 1),  batch];
hidIn = batch * params.wInpToHid;
hidOut = tanh(hidIn);

hidOut = [ones(size(hidOut, 1), 1),  hidOut];
outIn = hidOut * params.wHidToOut;

centerProbOutputs = outIn(:, params.centerProbIndices);
maxValues = max(centerProbOutputs, [], 2);
netInput = bsxfun(@minus, centerProbOutputs, maxValues);
netInput = exp(netInput);
centersProbs = bsxfun(@rdivide, netInput, sum(netInput, 2)); 

centerLocOutputs = outIn(:, params.centerLocIndices);

centerVarOutputs = outIn(:, params.centerVarIndices);
centerVarOutputs = exp(centerVarOutputs);

centerLocs = reshape(centerLocOutputs, size(centerLocOutputs, 1), [], params.numCenters);
centerVars = reshape(centerVarOutputs, size(centerVarOutputs, 1), 1, params.numCenters);

output.input = batch;
output.hidden = hidOut;
output.centersProbs = centersProbs;
output.centerLocs = centerLocs;
output.centerVars = centerVars;

end

function [cost, grad] = ComputeCostAndLastLayerGradient(netOutput, target)

numCenters = size(netOutput.centersProbs, 2);
targetDim  = size(target, 2);
batchSize = size(target, 1);

probConst = 1/((2*pi)^(targetDim/2));

probAll = zeros(batchSize, numCenters);
norm_k = zeros(batchSize, numCenters);
diff_k = zeros(batchSize, targetDim, numCenters);
for k = 1 : numCenters
    var_k = netOutput.centerVars(:, 1, k);
    center_k = netOutput.centerLocs(:, :, k);
    prob_k =  netOutput.centersProbs(:, k);
    diff_k(:, :, k)  = center_k - target;
    norm_k(:, k)  = sum(diff_k(:, :, k) .^ 2, 2);
    
    prob = probConst * (1 ./ (var_k .^ (targetDim))) .*  exp( - norm_k(:, k) ./ (2 * (var_k .^ 2)));   
    prob = prob .* prob_k;
    probAll(:, k) = prob;
end

probMargin = sum(probAll, 2);
LogProbMargin = log(probMargin);
cost = -sum(LogProbMargin);

posterior =  bsxfun(@rdivide, probAll, probMargin);

probGrad = netOutput.centersProbs - posterior;
sigmaGrad = - posterior .* (norm_k ./ (squeeze(netOutput.centerVars) .^ 2) - targetDim);
muGrad = zeros(size(netOutput.centerLocs));
for k=1:numCenters
    muGrad(:, :, k) = diff_k(:, :, k) .* repmat(posterior(:, k), 1, targetDim) ./ repmat(netOutput.centerVars(:, 1, k) .^ 2, 1, targetDim);
end

grad = [probGrad, reshape(muGrad, batchSize, []), sigmaGrad];

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

function params = Train(inputBatch, target, params)

lr = 0.01;
for iter = 1:100000
    output = ForwardPass(inputBatch, params);
    [cost, grad] = ComputeCostAndLastLayerGradient(output, target);
    fprintf('The cost is %f \n', cost);
    
    gradInfo = BackPropagate(grad, output, params);
    params = UpdateParams(gradInfo, params, lr);
    
    if(iter < 30000)
        lr  = lr * 0.9998;
    end     
end

end

function output = GetMeanAndSTD(t, params)

output = ForwardPass(t, params);
numCenters = size(output.centersProbs, 2);
targetDim  = size(output.centerLocs, 2);

batchSize = size(output.input, 1);

predMean = zeros(batchSize, targetDim);
for i = 1:numCenters 
    predMean = predMean + bsxfun(@times, output.centerLocs(:, :, i), output.centersProbs(:, i));
end

sigma = 0;
for i = 1:numCenters 
    sigma = sigma + (sum((output.centerLocs(:, :, i) - predMean) .^ 2, 2) + (output.centerVars(:, :, i) .^ 2)) .* output.centersProbs(:, i);
end
sigma = sqrt(sigma);

[~, maxCompInd] = max(output.centersProbs, [], 2);
predMaxProb = zeros(batchSize, targetDim);
for i = 1:numCenters 
    predMaxProb(maxCompInd == i, :) = output.centerLocs(maxCompInd == i, :, i);    
end

output.predMean = predMean;
output.predSigma = sigma;
output.predMaxComp = predMaxProb;


end



