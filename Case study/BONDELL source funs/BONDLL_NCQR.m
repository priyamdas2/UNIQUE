function results = BONDLL_NCQR(X, Y, tau_grid, opts)
% =========================================================================
% BONDLL_NCQR
%
% Bondell-style joint noncrossing linear quantile regression on [0,1]^p.
%
% The method jointly fits quantiles over tau_grid by solving a single LP:
%
%   min_{beta0,beta,s,u,v}  sum_{k=1}^K sum_{i=1}^n [ tau_k*u_{ik} + (1-tau_k)*v_{ik} ]
%
% subject to
%   y_i - beta0_k - x_i' beta_k = u_{ik} - v_{ik},      u_{ik},v_{ik} >= 0
%
% and Bondell-style noncrossing constraints on x in [0,1]^p:
%   (beta0_k - beta0_{k-1}) + sum_j (beta_{jk}-beta_{j,k-1}) x_j >= 0
%   for all x in [0,1]^p, k = 2,...,K.
%
% This is enforced via auxiliary variables s_{jk} >= 0:
%   s_{jk} >= -(beta_{jk} - beta_{j,k-1})
%   (beta0_k - beta0_{k-1}) - sum_j s_{jk} >= 0
%
% INPUTS
%   X         : n x p design matrix
%   Y         : n x 1 response vector
%   tau_grid  : 1 x K or K x 1 vector of quantile levels
%   opts      : optional struct
%
% Recommended opts fields:
%   opts.verbose         : 0/1, default 1
%   opts.scale_X_to_unit : 0/1, default 1
%   opts.linprog_display : 'none','final','iter', default 'none'
%   opts.linprog_options : optimoptions('linprog',...) object (optional)
%
% OUTPUT
%   results : struct, intentionally aligned with UNIQUE-style outputs
% =========================================================================

tic;

% -------------------------------------------------------------------------
% Basic checks
% -------------------------------------------------------------------------
if nargin < 4 || isempty(opts)
    opts = struct();
end

[n, p] = size(X);
Y = Y(:);
tau_grid = tau_grid(:)';
K = numel(tau_grid);

if length(Y) ~= n
    error('Length of Y must match number of rows of X.');
end

if any(tau_grid <= 0 | tau_grid >= 1)
    error('All entries of tau_grid must lie strictly between 0 and 1.');
end

opts = fill_default_opts_bondell(opts);

% -------------------------------------------------------------------------
% SAFE GUARD: scale X to [0,1] if needed
% -------------------------------------------------------------------------
tol_x = 1e-12;

X_min   = min(X, [], 1);
X_max   = max(X, [], 1);
X_range = X_max - X_min;

outside_mask = (X_min < -tol_x) | (X_max > 1 + tol_x);
did_scale_X = any(outside_mask) && opts.scale_X_to_unit;

X_input_original = X;

if did_scale_X
    if opts.verbose == 1
        fprintf('\nWarning: Some covariates lie outside [0,1].\n');
        fprintf('Bondell fit will linearly scale X into [0,1]^p before optimization.\n');
    end

    X_scaled = X;
    for j = 1:p
        if X_range(j) > tol_x
            X_scaled(:,j) = (X(:,j) - X_min(j)) ./ X_range(j);
        else
            X_scaled(:,j) = 0;   % constant column
        end
    end
    X = X_scaled;
end

% -------------------------------------------------------------------------
% Variable indexing
% Decision vector z = [beta0 ; beta ; s ; u ; v]
% -------------------------------------------------------------------------
n_beta0 = K;
n_beta  = p * K;
n_s     = p * (K - 1);
n_u     = n * K;
n_v     = n * K;

n_var = n_beta0 + n_beta + n_s + n_u + n_v;

idx_beta0_start = 1;
idx_beta0_end   = idx_beta0_start + n_beta0 - 1;

idx_beta_start  = idx_beta0_end + 1;
idx_beta_end    = idx_beta_start + n_beta - 1;

idx_s_start     = idx_beta_end + 1;
idx_s_end       = idx_s_start + n_s - 1;

idx_u_start     = idx_s_end + 1;
idx_u_end       = idx_u_start + n_u - 1;

idx_v_start     = idx_u_end + 1;
idx_v_end       = idx_v_start + n_v - 1;

if idx_v_end ~= n_var
    error('Index construction error: final index does not match n_var.');
end

beta0_idx = @(k) idx_beta0_start + (k - 1);
beta_idx  = @(j,k) idx_beta_start + (k - 1) * p + (j - 1);
s_idx     = @(j,k) idx_s_start    + (k - 2) * p + (j - 1);   % k = 2,...,K
u_idx     = @(i,k) idx_u_start    + (k - 1) * n + (i - 1);
v_idx     = @(i,k) idx_v_start    + (k - 1) * n + (i - 1);
% -------------------------------------------------------------------------
% Objective: sum tau*u + (1-tau)*v
% -------------------------------------------------------------------------
f = zeros(n_var, 1);

for k = 1:K
    tau_k = tau_grid(k);
    for i = 1:n
        f(u_idx(i,k)) = tau_k;
        f(v_idx(i,k)) = 1 - tau_k;
    end
end

% -------------------------------------------------------------------------
% Equality constraints:
% beta0_k + x_i' beta_k + u_ik - v_ik = y_i
% -------------------------------------------------------------------------
n_eq = n * K;
nnz_eq_est = n * K * (p + 3);

Ieq = zeros(nnz_eq_est, 1);
Jeq = zeros(nnz_eq_est, 1);
Veq = zeros(nnz_eq_est, 1);
beq = zeros(n_eq, 1);

ptr = 0;
row = 0;

for k = 1:K
    for i = 1:n
        row = row + 1;

        % beta0_k
        ptr = ptr + 1;
        Ieq(ptr) = row;
        Jeq(ptr) = beta0_idx(k);
        Veq(ptr) = 1;

        % beta(:,k)
        for j = 1:p
            ptr = ptr + 1;
            Ieq(ptr) = row;
            Jeq(ptr) = beta_idx(j,k);
            Veq(ptr) = X(i,j);
        end

        % u_ik
        ptr = ptr + 1;
        Ieq(ptr) = row;
        Jeq(ptr) = u_idx(i,k);
        Veq(ptr) = 1;

        % v_ik
        ptr = ptr + 1;
        Ieq(ptr) = row;
        Jeq(ptr) = v_idx(i,k);
        Veq(ptr) = -1;

        beq(row) = Y(i);
    end
end

Aeq = sparse(Ieq(1:ptr), Jeq(1:ptr), Veq(1:ptr), n_eq, n_var);

% -------------------------------------------------------------------------
% Inequality constraints for Bondell noncrossing on [0,1]^p:
%
% For k = 2,...,K, j = 1,...,p:
%   s_{jk} >= -(beta_{jk} - beta_{j,k-1})
%   <=> -s_{jk} - beta_{jk} + beta_{j,k-1} <= 0
%
% For k = 2,...,K:
%   (beta0_k - beta0_{k-1}) - sum_j s_{jk} >= 0
%   <=> beta0_{k-1} - beta0_k + sum_j s_{jk} <= 0
% -------------------------------------------------------------------------
n_ineq = p * (K - 1) + (K - 1);

nnz_ineq_est = 3 * p * (K - 1) + (p + 2) * (K - 1);
Iin = zeros(nnz_ineq_est, 1);
Jin = zeros(nnz_ineq_est, 1);
Vin = zeros(nnz_ineq_est, 1);
bin = zeros(n_ineq, 1);

ptr = 0;
row = 0;

% slope-difference rows
for k = 2:K
    for j = 1:p
        row = row + 1;

        % + beta_{j,k-1}
        ptr = ptr + 1;
        Iin(ptr) = row;
        Jin(ptr) = beta_idx(j, k-1);
        Vin(ptr) = 1;

        % - beta_{j,k}
        ptr = ptr + 1;
        Iin(ptr) = row;
        Jin(ptr) = beta_idx(j, k);
        Vin(ptr) = -1;

        % - s_{j,k}
        ptr = ptr + 1;
        Iin(ptr) = row;
        Jin(ptr) = s_idx(j, k);
        Vin(ptr) = -1;

        bin(row) = 0;
    end
end

% intercept rows
for k = 2:K
    row = row + 1;

    % + beta0_{k-1}
    ptr = ptr + 1;
    Iin(ptr) = row;
    Jin(ptr) = beta0_idx(k-1);
    Vin(ptr) = 1;

    % - beta0_k
    ptr = ptr + 1;
    Iin(ptr) = row;
    Jin(ptr) = beta0_idx(k);
    Vin(ptr) = -1;

    % + sum_j s_{j,k}
    for j = 1:p
        ptr = ptr + 1;
        Iin(ptr) = row;
        Jin(ptr) = s_idx(j, k);
        Vin(ptr) = 1;
    end

    bin(row) = 0;
end

A = sparse(Iin(1:ptr), Jin(1:ptr), Vin(1:ptr), n_ineq, n_var);
b = bin;

% -------------------------------------------------------------------------
% Bounds
% beta0, beta are free
% s, u, v are nonnegative
% -------------------------------------------------------------------------
lb = -inf(n_var, 1);
ub =  inf(n_var, 1);

if n_s > 0
    lb(idx_s_start : idx_s_start + n_s - 1) = 0;
end
lb(idx_u_start : idx_u_start + n_u - 1) = 0;
lb(idx_v_start : idx_v_start + n_v - 1) = 0;

% -------------------------------------------------------------------------
% Solve LP
% -------------------------------------------------------------------------
if opts.verbose == 1
    fprintf('\n *** Running BONDLL_NCQR: joint noncrossing LP fit ...\n');
    fprintf('n = %d, p = %d, K = %d\n', n, p, K);
end

if isempty(opts.linprog_options)
    lp_options = optimoptions('linprog', 'Display', opts.linprog_display);
else
    lp_options = opts.linprog_options;
end

[z_hat, fval, exitflag, output] = linprog(f, A, b, Aeq, beq, lb, ub, lp_options);

time_bondell = toc;

if exitflag <= 0
    warning('linprog did not report successful termination. exitflag = %d', exitflag);
end

% -------------------------------------------------------------------------
% Extract coefficients
% -------------------------------------------------------------------------
beta0_hat_bondell = zeros(K,1);
beta_hat_bondell  = zeros(p,K);
s_hat             = zeros(p, max(K-1,0));

for k = 1:K
    beta0_hat_bondell(k) = z_hat(beta0_idx(k));
    for j = 1:p
        beta_hat_bondell(j,k) = z_hat(beta_idx(j,k));
    end
end

for k = 2:K
    for j = 1:p
        s_hat(j,k-1) = z_hat(s_idx(j,k));
    end
end

% -------------------------------------------------------------------------
% Back-transform to original X scale, if X was scaled
% -------------------------------------------------------------------------
if did_scale_X
    beta0_hat_scaled = beta0_hat_bondell;
    beta_hat_scaled  = beta_hat_bondell;

    beta0_hat_orig = zeros(K,1);
    beta_hat_orig  = zeros(p,K);

    inv_range = zeros(1,p);
    nz = X_range > tol_x;
    inv_range(nz) = 1 ./ X_range(nz);

    for k = 1:K
        beta_hat_orig(:,k) = beta_hat_scaled(:,k) .* inv_range(:);

        beta0_hat_orig(k) = beta0_hat_scaled(k) ...
            - sum(beta_hat_scaled(:,k)' .* (X_min .* inv_range));
    end

    beta0_hat_bondell = beta0_hat_orig;
    beta_hat_bondell  = beta_hat_orig;
end

% -------------------------------------------------------------------------
% Deployed predictor
% -------------------------------------------------------------------------
Qhat = @(x) ones(size(x,1),1) * beta0_hat_bondell(:)' + x * beta_hat_bondell;

q_pred_all = Qhat(X_input_original);
if K >= 2
    min_gap = min(min(diff(q_pred_all, 1, 2)));
else
    min_gap = NaN;
end

% Training check loss on observed data
check_loss = joint_check_loss_linear(beta0_hat_bondell, beta_hat_bondell, ...
                                     X_input_original, Y, tau_grid);

if opts.verbose == 1
    fprintf('Bondell LP finished. exitflag = %d\n', exitflag);
    fprintf('Objective value          = %.6e\n', fval);
    fprintf('Observed-data check loss = %.6e\n', check_loss);
    if K >= 2
        fprintf('Sanity check on observed X: min adjacent gap = %.3e\n', min_gap);
    end
    fprintf('Bondell computation time = %.1f sec\n', time_bondell);
end

% -------------------------------------------------------------------------
% Store results
% -------------------------------------------------------------------------
results = struct();

results.time_bondell = time_bondell;
results.X = X;
results.Y = Y;
results.tau_grid = tau_grid;
results.opts = opts;

results.beta0_hat_bondell = beta0_hat_bondell;
results.beta_hat_bondell  = beta_hat_bondell;
results.s_hat             = s_hat;

results.Qhat = Qhat;
results.min_gap = min_gap;
results.check_loss = check_loss;

results.fval = fval;
results.exitflag = exitflag;
results.output = output;

results.X_original = X_input_original;
results.did_scale_X = did_scale_X;
results.X_min = X_min;
results.X_max = X_max;
results.X_range = X_range;
results.X_used_for_fitting = X;

results.q_pred_all = q_pred_all;
results.z_hat = z_hat;

end

% =========================================================================
% Helper: defaults
% =========================================================================
function opts = fill_default_opts_bondell(opts)

if ~isfield(opts, 'verbose') || isempty(opts.verbose)
    opts.verbose = 1;
end

if ~isfield(opts, 'scale_X_to_unit') || isempty(opts.scale_X_to_unit)
    opts.scale_X_to_unit = 1;
end

if ~isfield(opts, 'linprog_display') || isempty(opts.linprog_display)
    opts.linprog_display = 'none';
end

if ~isfield(opts, 'linprog_options')
    opts.linprog_options = [];
end

end

% =========================================================================
% Helper: joint check loss for linear quantile model
% =========================================================================
function loss = joint_check_loss_linear(beta0, beta, X, Y, tau_grid)

[n, ~] = size(X);
K = numel(tau_grid);

loss = 0;
for k = 1:K
    r = Y - beta0(k) - X * beta(:,k);
    tau = tau_grid(k);
    loss = loss + sum(r .* (tau - (r < 0)));
end

loss = loss / n;

end