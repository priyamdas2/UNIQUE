function [beta0, beta] = qr_lasso_linprog(X, y, tau, lambda)
% ==========================================================
% Quantile regression with L1 penalty (LASSO) via LP (linprog)
%   min_{b0,b}  sum rho_tau(y - b0 - X b) + lambda * ||b||_1
% where rho_tau(r) = r*(tau - I(r<0)).
%
% b0 is NOT penalized.
% ==========================================================

[n, p] = size(X);
y = y(:);

if tau <= 0 || tau >= 1
    error('tau must lie strictly between 0 and 1.');
end
if lambda < 0
    error('lambda must be nonnegative.');
end

% Decision variables:
% z = [b0;
%      b_plus  (p);
%      b_minus (p);
%      u       (n);
%      v       (n)]
nvar = 1 + 2*p + 2*n;

% Indices
idx_b0 = 1;
idx_bplus_start  = 2;
idx_bminus_start = 2 + p;
idx_u_start      = 1 + 2*p + 1;
idx_v_start      = 1 + 2*p + n + 1;

% Objective
f = zeros(nvar,1);

% lambda * sum(b_plus + b_minus)
f(idx_bplus_start  : idx_bplus_start  + p - 1) = lambda;
f(idx_bminus_start : idx_bminus_start + p - 1) = lambda;

% tau*sum(u) + (1-tau)*sum(v)
f(idx_u_start : idx_u_start + n - 1) = tau;
f(idx_v_start : idx_v_start + n - 1) = 1 - tau;

% Equality constraints:
% b0*1 + X*b_plus - X*b_minus + u - v = y
Aeq = sparse(n, nvar);
beq = y;

Aeq(:, idx_b0) = 1;
Aeq(:, idx_bplus_start  : idx_bplus_start  + p - 1) = X;
Aeq(:, idx_bminus_start : idx_bminus_start + p - 1) = -X;
Aeq(:, idx_u_start : idx_u_start + n - 1) = speye(n);
Aeq(:, idx_v_start : idx_v_start + n - 1) = -speye(n);

% Bounds
lb = -inf(nvar,1);
ub =  inf(nvar,1);

lb(idx_bplus_start  : idx_bplus_start  + p - 1) = 0;
lb(idx_bminus_start : idx_bminus_start + p - 1) = 0;
lb(idx_u_start      : idx_u_start      + n - 1) = 0;
lb(idx_v_start      : idx_v_start      + n - 1) = 0;

% Solve LP
opts = optimoptions('linprog', 'Display', 'none', 'Algorithm', 'dual-simplex');
[z, ~, exitflag] = linprog(f, [], [], Aeq, beq, lb, ub, opts);

if exitflag <= 0 || isempty(z)
    error('qr_lasso_linprog failed: linprog did not converge.');
end

% Extract solution
beta0   = z(idx_b0);
b_plus  = z(idx_bplus_start  : idx_bplus_start  + p - 1);
b_minus = z(idx_bminus_start : idx_bminus_start + p - 1);
beta    = b_plus - b_minus;

end