%% MAIN script: runs CD_BW_lasso; SGD_lasso and SGD_Momentum_lasso
%  It generates a simulated dataset in order to test the accuracy of the
%  predictions, comparing with the MATLAB built-in function: "lasso".
% %  
% %                     
% %   Optimization HW3 (October, 2018)
% %   Harold A. Hernández Roig (hahernan@est-econ.uc3m.es)

% Simulated Data:
[X,Y, weights] = sample_data(100,200,10);

% Samller Simulation:
%[X,Y, weights] = sample_data(100,20,5);

% Batch size: 
% - in "CD_BW_lasso" sets the amount of components of "w" to update in each iteration.
% - in "SGD_lasso" and "SGD_Momentum_lasso" sets the 
b = 10;


%% CD_BW_lasso: (Block-Wise) Coordinate Descent Algoritm for Penalized (LASSO) 
%              Least Square Regression.

% b = 1; % Uncomment to obtain the component-wise Coordinate Descent!
[G, W, MSPE, min_MSPE, lambda_min_MSPE, it_updates, Matlab_ans, FitInfo] ...
    = CD_BW_lasso(Y,X,b);

opt_lambda = G(lambda_min_MSPE);
mean_updates_per_lambda = it_updates/length(G);

disp(['Coordinate Descent. Batch size b = ', num2str(b)])
disp(['Optimum lambda = ', num2str(opt_lambda), '. Minimum MSPE = ', num2str(min_MSPE), '. Approx. Updated per Lambda =', num2str(round(mean_updates_per_lambda))])

%% coor_desc_lasso: Cyclical (Block-Wise) Coordinate Descent Algoritm for 
%                   Penalized (LASSO) Least Square Regression. Usage of
%                   soft-trhesholding included.

% b = 1; % Uncomment to obtain the component-wise Coordinate Descent!
[G, W, MSPE, min_MSPE, lambda_min_MSPE, it_updates, Matlab_ans,FitInfo]...
    = coor_desc_lasso(Y,X,b);

opt_lambda = G(lambda_min_MSPE);
mean_updates_per_lambda = it_updates/length(G);

disp(['Coordinate Descent (Soft-Thresholding). Batch size b = ', num2str(b)])
disp(['Optimum lambda = ', num2str(opt_lambda), '. Minimum MSPE = ', num2str(min_MSPE), '. Approx. Updated per Lambda =', num2str(round(mean_updates_per_lambda))])


%% SGD_lasso: Stochastic Gradient Descent for Penalized (LASSO)  Least Square
%            Regression. 

% b = 1; % Uncomment to obtain the single-component Stochastic Gradient Descent!
[G, W, MSPE, min_MSPE, lambda_min_MSPE, it_updates, Matlab_ans, FitInfo] = ...
    SGD_lasso(Y,X,b);

opt_lambda = G(lambda_min_MSPE);
mean_updates_per_lambda = it_updates/length(G);

disp(['SGD. Batch size b = ', num2str(b)])
disp(['Optimum lambda = ', num2str(opt_lambda), '. Minimum MSPE = ', num2str(min_MSPE), '. Approx. Updated per Lambda =', num2str(round(mean_updates_per_lambda))])

%% SGD_Momentum_lasso: Stochastic Gradient Descent with Momentum for Penalized (LASSO) 
%                     Least Square Regression.

% b = 1; % Uncomment to obtain the single-component Stochastic Gradient Descent with Momentum!
[G, W, MSPE, min_MSPE, lambda_min_MSPE, it_updates, Matlab_ans, FitInfo] =...
    SGD_Momentum_lasso(Y,X,b);

opt_lambda = G(lambda_min_MSPE);
mean_updates_per_lambda = it_updates/length(G);


disp(['SGD With Momentum. Batch size b = ', num2str(b)])
disp(['Optimum lambda = ', num2str(opt_lambda), '. Minimum MSPE = ', num2str(min_MSPE), '. Approx. Updated per Lambda =', num2str(round(mean_updates_per_lambda))])
