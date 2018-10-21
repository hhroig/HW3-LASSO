function S = S_lambda(lambda, X)
% S_LAMBDA(X) = sign(X)*subplus(abs(X) - lambda)
% Soft-thresholding function for "coor_desc_lasso"
%                                        
%   Optimization HW3 (October, 2018)
%   Harold A. Hernández Roig (hahernan@est-econ.uc3m.es)
S = sign(X).*subplus(abs(X) - lambda*ones(size(X)));
end