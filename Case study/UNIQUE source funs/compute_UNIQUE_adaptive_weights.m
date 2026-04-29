function [beta0_hat_mult, beta_hat_mult, qfit_j, signal_score, omega] = ...
    compute_UNIQUE_adaptive_weights(X, Y, tau_grid, eps_w)
% =======================================================================
% compute_unilasso_adaptive_weights
%
% Computes multivariate QR coefficients, variable-level quantile fit scores,
% signal scores, and adaptive weights for UniLasso.
%
% INPUTS
%   X        : n x p design matrix
%   Y        : n x 1 response vector
%   tau_grid : 1 x K (or K x 1) vector of quantile levels
%   eps_w    : small positive stabilization constant for omega
%
% OUTPUTS
%   beta0_hat_mult : 1 x K multivariate QR intercepts
%   beta_hat_mult  : p x K multivariate QR slopes
%   qfit_j         : p x 1 variable-level pseudo-R1 scores
%   signal_score   : p x K signal score matrix
%   omega          : p x K adaptive penalty weights
%
% NOTES
%   - beta_hat_mult is used only for adaptive weights / signal scoring.
%   - This does NOT affect the UniLasso basis, which should still be built
%     from univariate QR fits.
% =======================================================================

if nargin < 4 || isempty(eps_w)
    eps_w = 1e-6;
end

[n, p] = size(X);
K = length(tau_grid);

%% =======================
%  MULTIVARIATE QR COEFFICIENTS for adaptive weights only
%% =======================
beta0_hat_mult = zeros(1, K);
beta_hat_mult  = zeros(p, K);

for k = 1:K
    tau = tau_grid(k);

    % Multivariate linear quantile regression:
    %   Q_Y(tau | X) = b0 + X * b
    [b0_mult, b_mult] = qr_linprog_multi(X, Y, tau);

    beta0_hat_mult(k)  = b0_mult;
    beta_hat_mult(:,k) = b_mult(:);
end

%% =======================
%  Variable-level quantile fit quality qfit_j
%% =======================
qfit_j = zeros(p, 1);

for j = 1:p
    xj = X(:,j);

    loss_fit_total  = 0;
    loss_null_total = 0;

    for k = 1:K
        tau = tau_grid(k);

        % Univariate QR fit for predictor j at quantile tau
        [b0_jk, b1_jk] = qr_linprog(xj, Y, tau);
        qhat_jk = b0_jk + b1_jk * xj;

        % Null model: intercept-only tau-quantile
        q0 = quantile(Y, tau);

        % Check losses
        u_fit  = Y - qhat_jk;
        u_null = Y - q0;

        rho_fit  = u_fit  .* (tau - (u_fit < 0));
        rho_null = u_null .* (tau - (u_null < 0));

        loss_fit_total  = loss_fit_total  + sum(rho_fit);
        loss_null_total = loss_null_total + sum(rho_null);
    end

    % Combined pseudo-fit score for variable j across all quantiles
    qfit_j(j) = max(0, 1 - loss_fit_total / max(loss_null_total, 1e-8));
end

%% =======================
%  Signal score + adaptive weights
%% =======================
% signal_score = abs(beta_hat_mult) .* qfit_j;
beta_scale = median(abs(beta_hat_mult(:)));
signal_score = (abs(beta_hat_mult) / max(beta_scale, 1e-8)) .* qfit_j;

omega = 1 ./ (signal_score + eps_w);

end