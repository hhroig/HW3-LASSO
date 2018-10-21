function [X,Y, weights] = sample_data(n,p,non_zero)
% SAMPLE_DATA: Simulates Data Set
% Inputs (with example values):
% n = 100; % Sample size
% p = 200; % Number of Predictors
% non_zero = 10; % Non-zero predictors
%                     
%   Optimization HW3 (October, 2018)
%   Harold A. Hernández Roig (hahernan@est-econ.uc3m.es)

rng default % For reproducibility
weights = zeros(p,1); % Only two nonzero coefficients
weights(randperm(p,non_zero)) = randi([-5 5],1,non_zero);
X = normalize(randn(n,p));
Y = X*weights + randn(n,1)*0.1; % Small added noise
end

