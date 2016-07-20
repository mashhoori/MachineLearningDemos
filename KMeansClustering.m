
function KMeansClustering()
    
    img = imread('image.jpg'); 
    imshow(img);
    
    
    mtrx = reshape(img, size(img, 1) *  size(img, 2), 3)';
    mtrx = double(mtrx);
    numCluster = 5;
    [code, centers] = Cluster_KMeans(mtrx, numCluster);
    
    newmtrx = centers(:, code);
    newmtrx = newmtrx';
    
    newImg = reshape(newmtrx , size(img, 1), size(img, 2), 3);
    newImg = uint8(newImg);
    
    imshow(newImg)
        
end



% X  p * n 
function [code, centers] = Cluster_KMeans(X, k)

numInstance =  size(X, 2);
initCenterIndices = randsample(numInstance, k, false);
centers = X(:, initCenterIndices);

for i = 1:10
    
    dist = zeros(k, numInstance);
    for j = 1:k    
        df = bsxfun(@minus, X, centers(:, j));  
        dist(j, :) = sum(df .^ 2);    
    end
    
    [~, code]  = min(dist);        
    for j = 1:k
        centers(:,  j) = mean(X(:, code == j), 2);
    end
end


end