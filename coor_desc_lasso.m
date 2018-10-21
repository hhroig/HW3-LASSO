function [G, W, MSPE, min_MSPE, lambda_min_MSPE, it_updates, Matlab_ans,FitInfo] = coor_desc_lasso(Y,X,b)
% COOR_DESC: Cyclical (Block-Wise) Coordinate Descent Algoritm for 
%           Penalized (LASSO) Least Square Regression. Usage of
%           soft-trhesholding included.
%   Outputs:
%   G: grid of lambdas
%   W: Matrix of estimated weights (betas of the regression)
%   MSPE: Mean Squares Error for each penalty parameter
%   min_MSPE: minimum MSPE
%   lambda_min_MSPE: location of the lambda penalty corresponding to min_MSPE 
%   it_updates: total updates achieved
%   Matlab_ans: Matrix of estimated weights obtained through MATLAB's bult-in "lasso"
%   FitInfo: Info. about model fitting through MATLAB's "lasso"
%                                        
%   Optimization HW3 (October, 2018)
%   Harold A. Hernández Roig (hahernan@est-econ.uc3m.es)

tic
[n,p] = size(X);

% Standardize Data (not needed for our simulated data): 
% X = normalize(X);
% Y = normalize(Y);

%% Initialize Parameter and Split Data into Train/Test Sets
% choose the maximum lambda such that betas = 0 is the only optimal sol.
% lambda_max = mx_j | 1/n <x_j, y > |
lambda_max = max(abs(1/n*Y'*X));

% G is the grid of lambdas:
quant_lambdas = 100;
G = linspace(lambda_max, 0, quant_lambdas);

train_ind = false(n,1);
train_ind(randperm(n, round(0.8*n))) = true;

x = X(train_ind,:); % train predictors
y = Y(train_ind);   % train responses

x_test = X(~train_ind,:);
y_test = Y(~train_ind);

N = length(y);      %number of observations in train set 
q = length(y_test); %number of observations in test set

MSPE = zeros(1,quant_lambdas); % meas squared predictive error

w = zeros(p,1); % we declare the weights here so they can be used inside the cycle as a "warm start"
W = 10*ones(p,quant_lambdas); % we record all the estimated weights

it_updates = 0;
for g_index = 1:quant_lambdas
    lambda = G(g_index);
     
    %% CD algorithm:    
    conv_diff = abs(W(:,g_index)-w); % we fix a "big" number to monitor the conv. error
    
    while max(conv_diff) > eps
    
    % repeatedly cycle through the predictors in some arbitrary
    % order J, with only 10% of predictors each time.
    J = randperm(p,b);
    
    for j = 1:length(J)
        r = y - x*w;        
        w(J(j)) = S_lambda(lambda,w(J(j)) + 1/N*x(:,J(j))'*r); % soft-thresholding operator
        it_updates = it_updates + 1;
    end % of w components updating
    
    conv_diff = abs(W(:,g_index) - w); % check convergence of the updated w
    W(:,g_index) = w;                  % save updated w
    end % of "while"  
    
    MSPE(g_index) = 1/q*( y_test - x_test*w )' * ( y_test - x_test*w );
end % of the lambda grid "for"

[min_MSPE,lambda_min_MSPE] = min(MSPE);
toc
% compare:
[Matlab_ans,FitInfo] = lasso(X,Y,'Lambda',G);
end % of "coor_desc_lasso" function

