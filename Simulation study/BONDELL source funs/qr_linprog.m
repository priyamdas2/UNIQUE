function [beta0, beta] = qr_linprog(X, y, tau)
% ==========================================================
% Plain quantile regression (no LASSO) via LP (linprog)
%   min_{b0,b}  sum rho_tau(y - b0 - X b)
% rho_tau(r)= r*(tau - I(r<0)).
%
% Requires Optimization Toolbox (linprog).
% ==========================================================

[n, p] = size(X);

% Variables:
% z = [b0; beta(p); u(n); v(n)]
% Equality: b0*1 + X*beta + u - v = y
% u >= 0, v >= 0, beta free

nvar = 1 + p + 2*n;

f = zeros(nvar,1);
idx_u = 1 + p + 1;
idx_v = 1 + p + n + 1;

f(idx_u:idx_u+n-1) = tau;
f(idx_v:idx_v+n-1) = (1 - tau);

Aeq = sparse(n, nvar);
beq = y;

Aeq(:,1) = 1;                 % b0
Aeq(:,2:1+p) = X;             % beta
Aeq(:,idx_u:idx_u+n-1) = speye(n);   % +u
Aeq(:,idx_v:idx_v+n-1) = -speye(n);  % -v

lb = -inf(nvar,1);
ub =  inf(nvar,1);
lb(idx_u:idx_u+n-1) = 0;
lb(idx_v:idx_v+n-1) = 0;

opts = optimoptions('linprog','Display','none','Algorithm','dual-simplex');
z = linprog(f, [], [], Aeq, beq, lb, ub, opts);

beta0 = z(1);
beta  = z(2:1+p);

end
