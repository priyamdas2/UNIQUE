function init = build_unilasso_feasible_init(X, Y, tau_grid, signal_score, ...
    beta0_univ_hat, beta_univ_hat, same_sign_mask, t_row, gate_thr, lb, ub, ...
    theta_cap, beta0_contrib_cap, verbose_stage_2_init)

[~, p] = size(X);
K = numel(tau_grid);

eps_sig = 1e-12;

% Normalize signal scores
S = signal_score / max(signal_score(:) + eps_sig);

% Row-level relevance
S_row = size(S,2) ./ sum(1 ./ max(S, 1e-8), 2); % Harmonic mean
g_row = max(0, (S_row - t_row) / (1 - t_row));
g_row = g_row .^ 1.0;

% Cell-level relevance
row_max = max(S, [], 2);
g_cell = S ./ max(row_max, eps_sig);
g_cell = g_cell .^ 1.0;  % row wise normalization, num rows = K, g_cell ∝ 1/K


% Combined gate
gate = g_row .* g_cell;

vals = gate(gate > (gate_thr*9/K)); % if less K, higher threshold, since g_cell values ∝ 1/K

if isempty(vals)
    scale = 1;
else
    scale = 1 / mean(vals);
end
scale = min(scale, 5);
gate_scaled = scale * gate;

active_rows = find(max(abs(gate_scaled), [], 2) > 1e-12);

if verbose_stage_2_init == 1
    fprintf('Active rows (%d): %s\n', numel(active_rows), ...
        strjoin(string(active_rows), ', '));
end

% Reduced multivariate QR
beta0_red = zeros(K,1);
beta_red  = zeros(p,K);

if isempty(active_rows)
    warning('\nNo active rows selected by gate_scaled; using intercept-only QR.');
    for k = 1:K
        beta0_red(k) = quantile(Y, tau_grid(k));
    end
else
    X_active = X(:, active_rows);
    
    for k = 1:K
        tau = tau_grid(k);
        [b0_tmp, b_tmp] = qr_linprog_multi(X_active, Y, tau);
        beta0_red(k) = b0_tmp;
        beta_red(active_rows, k) = b_tmp(:);
    end
end

% ---------------------------------------------------------
% Raw theta initialization
% ---------------------------------------------------------
theta_init_raw = zeros(p, K);
tol_div = 1e-6 * max(1, max(abs(beta_univ_hat(:))));

for k = 1:K
    for j = active_rows(:)'
        denom = beta_univ_hat(j,k);

        if abs(denom) < tol_div
            ratio_raw = 0;
        else
            ratio_raw = beta_red(j,k) / denom;
        end

        if same_sign_mask(j,k)
            % constrained to be nonnegative
            theta_init_raw(j,k) = max(0, ratio_raw);
        else
            % free-sign entry
            theta_init_raw(j,k) = ratio_raw;
        end
    end
end

% ---------------------------------------------------------
% Stabilize raw theta initialization before projection
% ---------------------------------------------------------

for j = 1:p
    for k = 1:K
        if same_sign_mask(j,k)
            theta_init_raw(j,k) = min(theta_init_raw(j,k), theta_cap);
            theta_init_raw(j,k) = max(theta_init_raw(j,k), 0);
        else
            theta_init_raw(j,k) = max(min(theta_init_raw(j,k), theta_cap), -theta_cap);
        end
    end
end


for j = 1:p
    for k = 1:K
        if abs(beta0_univ_hat(j,k)) > 1e-10
            theta_allowed = beta0_contrib_cap / abs(beta0_univ_hat(j,k));

            if same_sign_mask(j,k)
                theta_init_raw(j,k) = min(theta_init_raw(j,k), theta_allowed);
                theta_init_raw(j,k) = max(theta_init_raw(j,k), 0);
            else
                theta_init_raw(j,k) = max(min(theta_init_raw(j,k), theta_allowed), -theta_allowed);
            end
        end
    end
end

% ---------------------------------------------------------
% Raw theta0 initialization
% Must be computed after theta_init_raw is stabilized
% ---------------------------------------------------------
theta0_init_raw = zeros(K,1);
for k = 1:K
    theta0_init_raw(k) = beta0_red(k) ...
        - sum(theta_init_raw(:,k) .* beta0_univ_hat(:,k));
end

% Projection to feasible set
proj_opts = struct();
proj_opts.w_theta0 = 1;
proj_opts.w_theta  = 1;
proj_opts.w_s      = 1e-6;
proj_opts.margin   = 1e-10;
proj_opts.freeze_row_thresh = 0.01;

if ~isequal(size(same_sign_mask), [p, K])
    error('same_sign_mask must be p x K.');
end


[theta0_init, theta_init, s_init] = project_unilasso_init_qp( ...
    theta0_init_raw, theta_init_raw, beta0_univ_hat, beta_univ_hat, ...
    same_sign_mask, lb, ub, proj_opts);

x_init = [theta0_init; theta_init(:)];

if verbose_stage_2_init == 1
    fprintf('theta_init summary:\n');
    fprintf('  max(theta_init(:))    = %.4f\n', max(theta_init(:)));
    fprintf('  mean(theta_init(:))   = %.4f\n', mean(theta_init(:)));
    fprintf('  nnz(theta_init > 1e-8)= %d out of %d\n', ...
        nnz(theta_init > 1e-8), numel(theta_init));
end

box_check_tol = 1e-10;
if any(x_init < lb - box_check_tol) || any(x_init > ub + box_check_tol)
    error('> UniLasso initialization violates box bounds even after repair.');
end

% Hard feasibility check
A0     = build_A(p, K, beta0_univ_hat, same_sign_mask);
Gamma0 = build_Gamma(theta0_init, theta_init, beta_univ_hat);
Ag0    = A0 * Gamma0;

min_margin = min(Ag0);

feas_check_tol = 1e-7;
if min_margin < -feas_check_tol
    error('> UniLasso initialization infeasible even after repair.');
else
    if verbose_stage_2_init == 1
        fprintf('> Feasible UniLasso initialization confirmed.\n');
    end
end


if verbose_stage_2_init == 1
    fprintf('gate summary:\n');
    fprintf('  max(gate_scaled(:))   = %.4f\n', max(gate_scaled(:)));
    fprintf('  mean(gate_scaled(:))  = %.4f\n', mean(gate_scaled(:)));
    fprintf('  nnz(gate_scaled > 0)  = %d out of %d\n', ...
        nnz(gate_scaled > 0), numel(gate_scaled));
end

init = struct();
init.S = S;
init.S_row = S_row;
init.g_row = g_row;
init.g_cell = g_cell;
init.gate = gate;
init.gate_scaled = gate_scaled;
init.active_rows = active_rows;
init.beta0_red = beta0_red;
init.beta_red = beta_red;
init.theta0_init_raw = theta0_init_raw;
init.theta_init_raw = theta_init_raw;
init.theta0_init = theta0_init;
init.theta_init = theta_init;
init.s_init = s_init;
init.min_margin = min_margin;
end