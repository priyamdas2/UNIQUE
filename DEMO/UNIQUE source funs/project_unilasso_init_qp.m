function [theta0_proj, theta_proj, s_proj] = project_unilasso_init_qp( ...
    theta0_raw, theta_raw, beta0_uni_hat, beta_uni_hat, ...
    same_sign_mask, lb_x, ub_x, opts)
% =======================================================================
% Project (theta0, theta) jointly onto the full UniLasso feasible set
% by solving a convex quadratic program over z = (theta0, theta, s).
%
% This version additionally enforces the box bounds on
%   x = [theta0; theta(:)]
% so that the projected initialization is compatible with SAPS_NCQR.
%
% Optionally freezes rows of theta whose raw entries are all below a
% threshold, preserving sparsity of the initialization.
%
% INPUTS
%   theta0_raw      : K x 1 raw intercept initialization
%   theta_raw       : p x K raw theta initialization
%   beta0_uni_hat   : p x K matrix used in main noncrossing block
%   beta_uni_hat    : p x K matrix used in slack construction
%   same_sign_mask  : p x K logical mask; true means theta_{j,k} >= 0
%   lb_x            : lower bound for x = [theta0; theta(:)], length K+pK
%   ub_x            : upper bound for x = [theta0; theta(:)], length K+pK
%   opts            : optional struct with fields
%                     .w_theta0
%                     .w_theta
%                     .w_s
%                     .margin
%                     .freeze_row_thresh
%
% OUTPUTS
%   theta0_proj     : projected K x 1 intercept vector
%   theta_proj      : projected p x K theta matrix
%   s_proj          : projected p x (K-1) slack matrix
% =======================================================================

% -----------------------------------------------------------------------
% Default options
% -----------------------------------------------------------------------
if nargin < 8 || isempty(opts), opts = struct(); end
if ~isfield(opts, 'w_theta0'), opts.w_theta0 = 1; end
if ~isfield(opts, 'w_theta'),  opts.w_theta  = 1; end
if ~isfield(opts, 'w_s'),      opts.w_s      = 1e-6; end
if ~isfield(opts, 'margin'),   opts.margin   = 1e-6; end
if ~isfield(opts, 'freeze_row_thresh'), opts.freeze_row_thresh = []; end

% -----------------------------------------------------------------------
% Dimensions
% -----------------------------------------------------------------------
[p, K] = size(theta_raw);

n0 = K;          % number of theta0 variables
n1 = p*K;        % number of theta variables
n2 = p*(K-1);    % number of slack variables
d  = n0 + n1 + n2;

% -----------------------------------------------------------------------
% Basic checks
% -----------------------------------------------------------------------
if ~isequal(size(theta0_raw), [K, 1]) && ~(isvector(theta0_raw) && numel(theta0_raw) == K)
    error('theta0_raw must have length K.');
end
theta0_raw = theta0_raw(:);

if ~isequal(size(beta0_uni_hat), [p, K])
    error('beta0_uni_hat must be p x K.');
end

if ~isequal(size(beta_uni_hat), [p, K])
    error('beta_uni_hat must be p x K.');
end

if ~isequal(size(same_sign_mask), [p, K])
    error('same_sign_mask must be p x K.');
end

same_sign_mask = logical(same_sign_mask);

lb_x = lb_x(:);
ub_x = ub_x(:);

if length(lb_x) ~= n0 + n1 || length(ub_x) ~= n0 + n1
    error('lb_x and ub_x must both have length K + p*K.');
end

if any(lb_x > ub_x)
    error('Each component of lb_x must be <= corresponding component of ub_x.');
end

% -----------------------------------------------------------------------
% Raw slack induced by raw theta
% -----------------------------------------------------------------------
s_raw = zeros(p, K-1);
for k = 2:K
    for j = 1:p
        s_raw(j,k-1) = max(0, ...
            theta_raw(j,k-1)*beta_uni_hat(j,k-1) - ...
            theta_raw(j,k)*beta_uni_hat(j,k));
    end
end

z_raw = [theta0_raw(:); theta_raw(:); s_raw(:)];

% -----------------------------------------------------------------------
% Quadratic objective
% Minimize 0.5*(z-z_raw)' W (z-z_raw)
% Equivalent to:
%   min 0.5 z' H z + f' z
% with H = W, f = -W z_raw
% -----------------------------------------------------------------------
w = [opts.w_theta0 * ones(n0,1);
     opts.w_theta  * ones(n1,1);
     opts.w_s      * ones(n2,1)];

H = spdiags(w, 0, d, d);
f = -H * z_raw;
H = H + 1e-12 * speye(d);  % tiny ridge for numerical stability

% -----------------------------------------------------------------------
% Full inequality constraints Afull z >= margin
% Blocks:
%   1) selected theta >= 0
%   2) s >= 0
%   3) s + beta_k theta_k - beta_{k-1} theta_{k-1} >= 0
%   4) main noncrossing
% -----------------------------------------------------------------------
n_sign = nnz(same_sign_mask);
rows_total = n_sign + n2 + n2 + (K-1);

Afull = sparse(rows_total, d);
bfull = opts.margin * ones(rows_total, 1);

row = 0;

% ---------- Block 1: selected theta >= 0 ----------
for k = 1:K
    for j = 1:p
        if same_sign_mask(j,k)
            row = row + 1;
            col_theta = n0 + (k-1)*p + j;
            Afull(row, col_theta) = 1;
        end
    end
end

% ---------- Block 2: s >= 0 ----------
for k = 2:K
    for j = 1:p
        row = row + 1;
        col_s = n0 + n1 + (k-2)*p + j;
        Afull(row, col_s) = 1;
    end
end

% ---------- Block 3: s + beta_k theta_k - beta_{k-1} theta_{k-1} >= 0 ----------
for k = 2:K
    for j = 1:p
        row = row + 1;

        col_theta_k   = n0 + (k-1)*p + j;
        col_theta_km1 = n0 + (k-2)*p + j;
        col_s         = n0 + n1 + (k-2)*p + j;

        Afull(row, col_theta_k)   =  beta_uni_hat(j,k);
        Afull(row, col_theta_km1) = -beta_uni_hat(j,k-1);
        Afull(row, col_s)         =  1;
    end
end

% ---------- Block 4: main noncrossing ----------
for k = 2:K
    row = row + 1;

    % theta0_k - theta0_{k-1}
    Afull(row, k)   =  1;
    Afull(row, k-1) = -1;

    % theta contributions
    for j = 1:p
        col_theta_k   = n0 + (k-1)*p + j;
        col_theta_km1 = n0 + (k-2)*p + j;

        Afull(row, col_theta_k)   = Afull(row, col_theta_k)   + beta0_uni_hat(j,k);
        Afull(row, col_theta_km1) = Afull(row, col_theta_km1) - beta0_uni_hat(j,k-1);
    end

    % subtract slack terms
    for j = 1:p
        col_s = n0 + n1 + (k-2)*p + j;
        Afull(row, col_s) = -1;
    end
end

% Convert Afull z >= bfull into quadprog form Aineq z <= bineq
Aineq = -Afull;
bineq = -bfull;

% -----------------------------------------------------------------------
% Equality constraints: freeze inactive rows of theta if requested
% -----------------------------------------------------------------------
Aeq = [];
beq = [];

if ~isempty(opts.freeze_row_thresh)
    inactive_rows = find(max(theta_raw, [], 2) < opts.freeze_row_thresh);

    n_eq = numel(inactive_rows) * K;
    if n_eq > 0
        Aeq = sparse(n_eq, d);
        beq = zeros(n_eq, 1);

        eqrow = 0;
        for rr = 1:numel(inactive_rows)
            j = inactive_rows(rr);
            for k = 1:K
                eqrow = eqrow + 1;
                col_theta = n0 + (k-1)*p + j;
                Aeq(eqrow, col_theta) = 1;
                beq(eqrow) = theta_raw(j,k);
            end
        end
    end
end

% -----------------------------------------------------------------------
% Box constraints for z = [theta0; theta; s]
%
% The SAPS box bounds are defined only for
%   x = [theta0; theta(:)].
%
% We extend them to z by imposing:
%   - theta0, theta use the supplied x-bounds
%   - slack s is constrained only to be nonnegative
%
% Note:
%   Block 2 already enforces s >= margin, but setting lb_z for slack to 0
%   is harmless and improves numerical stability.
% -----------------------------------------------------------------------
lb_z = [lb_x; zeros(n2,1)];
ub_z = [ub_x; inf(n2,1)];

if length(lb_z) ~= d || length(ub_z) ~= d
    error('Internal error: bound vectors lb_z / ub_z have incorrect length.');
end

if any(lb_z > ub_z)
    error('Internal error: some lower bounds exceed upper bounds in z-space.');
end

% -----------------------------------------------------------------------
% Solve QP
% -----------------------------------------------------------------------
qp_opts = optimoptions('quadprog', ...
    'Display', 'off', ...
    'Algorithm', 'interior-point-convex');

[z_proj, ~, exitflag] = quadprog(H, f, Aineq, bineq, Aeq, beq, lb_z, ub_z, [], qp_opts);

if exitflag <= 0
    error('project_unilasso_init_qp: quadprog failed with exitflag = %d', exitflag);
end

% -----------------------------------------------------------------------
% Extract projected variables
% -----------------------------------------------------------------------
theta0_proj = z_proj(1:n0);
theta_proj  = reshape(z_proj(n0+1:n0+n1), p, K);
s_proj      = reshape(z_proj(n0+n1+1:end), p, K-1);

% -----------------------------------------------------------------------
% Final safety checks
% -----------------------------------------------------------------------
x_proj = [theta0_proj; theta_proj(:)];

box_tol = 1e-10;
if any(x_proj < lb_x - box_tol) || any(x_proj > ub_x + box_tol)
    error('project_unilasso_init_qp: projected x violates box bounds.');
end

end