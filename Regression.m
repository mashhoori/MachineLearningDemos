%
% Demos for regression, resgression with regularization, and cross
% validation
% Ali Mashhoori. July 2016


clear all;
close all;

%%
f1 = @(x) (x .^ 4) - 2 * (x .^ 2) + rand(size(x));

x = -2:0.05:2;
y = f1(x);
dataset = [x; y];
%plot(x, y, 'o');

%%

trainset = dataset(:, 1:3:end);
testset = dataset(:, 2:3:end);

plot(trainset(1, :), trainset(2, :), 'o');
hold on
plot(testset(1, :),  testset(2, :), '*');
legend('Training Data', 'Test Data');
set (gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
pause(3)
close

x_train  = trainset(1, :);
y_train  = trainset(2, :);
x_test   = testset(1, :);
y_test   = testset(2, :);

%% 
%  Simple regression

orders = [1, 2, 3, 4, 5, 6, 10, 15, 20];
for po = orders

    orderOfPolynomial = po;

    X = zeros(length(x_train), orderOfPolynomial + 1);
    for i = 0:orderOfPolynomial
       X(:, i+1) = x_train .^ i;
    end
    coeff = (X' * X) \ X' * y_train';

    y_predicted_train = polyval(flip(coeff), x_train);
    y_predicted_test = polyval(flip(coeff), x_test);

    mse_train  = sum((y_train - y_predicted_train) .^ 2) / length(y_train);
    mse_test  = sum((y_test - y_predicted_test) .^ 2) / length(y_test);

    subplot(2, 1, 1)
    plot(x_train, y_train, 'ob');
    hold on
    plot(x_train, y_predicted_train, '-*r');
    title(['MSE for the training set is: '  num2str(mse_train)])
    legend('Target', 'Predicted')

    subplot(2, 1, 2)
    plot(x_test, y_test, 'ob');
    hold on
    plot(x_test, y_predicted_test, '-*r');
    title(['MSE for the test set is: '  num2str(mse_test)])
    legend('Target', 'Predicted')

    set(gcf,'Name',['Polynomial of order ' num2str(orderOfPolynomial)]);
    set (gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
        
    pause(3);
    close;    
end

%%  
%  Regression with regularization

landa = 10;  % regularization coefficient

orders = [1, 2, 3, 4, 5, 6, 10, 15, 20]; 
for po = orders
    
    orderOfPolynomial = po;
    
    X = zeros(length(x_train), orderOfPolynomial + 1);
    for i = 0:orderOfPolynomial
       X(:, i+1) = x_train .^ i;
    end    
    
    regTerm = eye(size(X, 2)) * landa;
    regTerm(1, 1) = 0;
    coeff = (X' * X + regTerm) \ X' * y_train';

    y_predicted_train = polyval(flip(coeff), x_train);
    y_predicted_test = polyval(flip(coeff), x_test);

    mse_train  = sum((y_train - y_predicted_train) .^ 2) / length(y_train);
    mse_test  = sum((y_test - y_predicted_test) .^ 2) / length(y_test);

    subplot(2, 1, 1)
    plot(x_train, y_train, 'ob');
    hold on
    plot(x_train, y_predicted_train, '-*r');
    title(['MSE for the training set is: '  num2str(mse_train)])
    legend('Target', 'Predicted')

    subplot(2, 1, 2)
    plot(x_test, y_test, 'ob');
    hold on
    plot(x_test, y_predicted_test, '-*r');
    title(['MSE for the test set is: '  num2str(mse_test)])
    legend('Target', 'Predicted')

    set(gcf,'Name',['Polynomial of order ' num2str(orderOfPolynomial)]);
    set(gcf, 'Units', 'normalized', 'Position', [0,0,1,1]);
        
    pause(3);
    close;    
end

%% Determining the best degree for the  fitting polynomial using cross validation

ind = randperm(numel(x_train));

x_train = x_train(ind);
y_train = y_train(ind);

numberOfFolds = 2;
orders = [1, 2, 3, 4, 5, 6, 10, 15, 20];

mse_orders = zeros(1, numel(orders));
for oi = 1:numel(orders)
    
    mse_valid_fold = zeros(1, numel(numberOfFolds));
    orderOfPolynomial = orders(oi);   
    
    for fold = 1:numberOfFolds          
        nInstance = floor(length(x_train) / numberOfFolds);
        sInd = (fold - 1) * nInstance + 1;
        eInd = min(length(x_train), fold * nInstance);
        
        validIndices = sInd:eInd;
        trainIndices = setdiff(1:length(x_train), validIndices);
        
        x_train_fold = x_train(trainIndices);
        y_train_fold = y_train(trainIndices);  
        x_valid_fold = x_train(validIndices);
        y_valid_fold = y_train(validIndices);        
                
        X = zeros(length(x_train_fold), orderOfPolynomial + 1);
        for i = 0:orderOfPolynomial
           X(:, i+1) = x_train_fold .^ i;
        end  
        coeff = (X' * X) \ X' * y_train_fold';
        
        y_predicted_valid = polyval(flip(coeff), x_valid_fold);
        mse_valid_fold(fold)  = sum((y_valid_fold - y_predicted_valid) .^ 2) / length(y_valid_fold);
        
        pause(3);
        close;            
    end
    
    mse_orders(oi) = mean(mse_valid_fold);    
end

[~, bestOrderIndex] = min(mse_orders);
bestOrder = orders(bestOrderIndex);
fprintf('Best validation mse was achieved using a polynomial of order %d \n', bestOrder);





