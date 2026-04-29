function [b0, b] = qr_linprog_multi(X, y, tau)
% ========================================================================
% qr_linprog_multi
%
% Multivariate linear quantile regression via linear programming:
%
%   min_{b0,b} sum_i rho_tau(y_i - b0 - x_i' b)
%
% INPUTS
%   X   : n x p design matrix
%   y   : n x 1 response
%   tau : quantile level in (0,1)
%
% OUTPUTS
%   b0  : intercept
%   b   : p x 1 slope vector
% ========================================================================

[n, p] = size(X);

% Variable ordering:
%   z = [b0; b_plus(p); b_minus(p); u_plus(n); u_minus(n)]
%
% where b = b_plus - b_minus, with b_plus, b_minus >= 0
% and residuals satisfy:
%   y - b0 - Xb = u_plus - u_minus
% with u_plus, u_minus >= 0

nvar = 1 + p + p + n + n;

idx_b0      = 1;
idx_bplus   = 2:(1+p);
idx_bminus  = (2+p):(1+2*p);
idx_uplus   = (2+2*p):(1+2*p+n);
idx_uminus  = (2+2*p+n):(1+2*p+2*n);

% Objective:
% minimize tau * sum(u_plus) + (1-tau) * sum(u_minus)
f = zeros(nvar,1);
f(idx_uplus)  = tau;
f(idx_uminus) = 1 - tau;

% Equality constraints:
%   b0 + X*(b_plus - b_minus) + u_plus - u_minus = y
Aeq = zeros(n, nvar);
Aeq(:, idx_b0)     = 1;
Aeq(:, idx_bplus)  = X;
Aeq(:, idx_bminus) = -X;
Aeq(:, idx_uplus)  = eye(n);
Aeq(:, idx_uminus) = -eye(n);

beq = y(:);

% Bounds:
% b0 is free, all others nonnegative
lb = zeros(nvar,1);
ub = inf(nvar,1);

lb(idx_b0) = -inf;
ub(idx_b0) = inf;

opts = optimoptions('linprog', ...
    'Algorithm', 'dual-simplex', ...
    'Display', 'none');

z = linprog(f, [], [], Aeq, beq, lb, ub, opts);

if isempty(z)
    error('qr_linprog_multi failed to find a solution.');
end

b0 = z(idx_b0);
b  = z(idx_bplus) - z(idx_bminus);

end