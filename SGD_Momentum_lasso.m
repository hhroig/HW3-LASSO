function [G, W, MSPE, min_MSPE, lambda_min_MSPE, it_updates, Matlab_ans, FitInfo] = SGD_Momentum_lasso(Y,X,b)
% SGD_Momentum_lasso: Stochastic Gradient Descent with Momentum for Penalized (LASSO) 
%                     Least Square Regression.
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
W = ones(p,quant_lambdas); % we record all the estimated weights

eta = 0.1; % Momentum parameter

it_updates = 0;
for g_index = 1:quant_lambdas
    lambda = G(g_index);
     
    %% CD algorithm:    
    conv_diff = abs(W(:,g_index)-w); % we fix a "big" number to step-in while
    
    alpha = 0.99; % initial learning-rate
    k = 1;
    temp_g_k = w;
    
    while max(conv_diff) > 10^(-16)  
    
    % repeatedly compute the gradient for "b" random observations:    
    rand_obs = randperm(N,b);
    g_k = 1/b*( -1/N*x(rand_obs,:)'*y(rand_obs) + 1/N*x(rand_obs,:)'*x(rand_obs,:)*w + 1/N*lambda*sign(w));% + 1/N*S_lambda(lambda,w) ; %
    
    v_k = g_k + eta*temp_g_k;
    
    w = w - alpha*v_k; % soft-thresholding operator        
    
    temp_g_k = g_k;
    
    conv_diff = abs(W(:,g_index) - w); % check convergence of the updated w
    W(:,g_index) = w;                  % save updated w
    
    k = k+1;
    alpha = alpha^(1.01); % other option, but faster decreasing: alpha = alpha/sqrt(k);
    
    it_updates = it_updates + 1;
    end % of "while"  
 
    MSPE(g_index) = 1/q*( y_test - x_test*w )' * ( y_test - x_test*w );
end % of the lambda grid "for"

[min_MSPE,lambda_min_MSPE] = min(MSPE);
toc
% compare:
[Matlab_ans,FitInfo] = lasso(X,Y,'Lambda',G);
end % of SGD_Momentum function

