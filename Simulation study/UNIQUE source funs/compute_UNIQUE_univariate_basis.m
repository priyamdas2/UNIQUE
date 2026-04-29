function [beta0_uni_hat, beta_uni_hat] = ...
    compute_UNIQUE_univariate_basis(X, Y, tau_grid)
% =======================================================================
% compute_unilasso_univariate_basis
%
% Computes univariate quantile regression coefficients for each predictor
% across all quantiles. These define the UniLasso basis.
%
% INPUTS
%   X        : n x p design matrix
%   Y        : n x 1 response vector
%   tau_grid : 1 x K (or K x 1) vector of quantiles
%
% OUTPUTS
%   beta0_uni_hat : p x K intercepts from univariate QR
%   beta_uni_hat  : p x K slopes from univariate QR
%
% NOTES
%   - These are the "basis functions" for UniLasso
%   - Must be consistent everywhere (constraints, Gamma, etc.)
% =======================================================================

[n, p] = size(X);
K = length(tau_grid);

beta0_uni_hat = zeros(p, K);
beta_uni_hat  = zeros(p, K);

for j = 1:p
    xj = X(:,j);

    for k = 1:K
        tau = tau_grid(k);

        [b0, b1] = qr_linprog(xj, Y, tau);

        beta0_uni_hat(j,k) = b0;
        beta_uni_hat(j,k)  = b1;
    end
end

end