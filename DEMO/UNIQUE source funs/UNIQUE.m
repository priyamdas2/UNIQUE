function results = UNIQUE(X, Y, tau_grid, opts)
% =========================================================================
% run_unilasso_qr_pipeline
%
% Full QR-UniLasso pipeline:
%   1) full-sample univariate QR basis
%   2) grouped cross-fitted meta-features eta_CF
%   3) adaptive weights
%   4) QR-LASSO initialization (optional)
%   5) feasible UniLasso initialization
%   6) outer train/test split for lambda tuning
%   7) SAPS_NCQR path
%   8) selected lambda
%   9) final full-data refit
%  10) deployed coefficients + evaluation (optional)
%
% INPUTS
%   X         : n x p
%   Y         : n x 1
%   tau_grid  : 1 x K
%   opts      : struct
%
% OUTPUT
%   results   : struct with fitted objects
% =========================================================================
tic;

[n, p] = size(X);
K = numel(tau_grid);

opts = fill_default_opts(opts, p, K);

% =========================================================
% SAFE GUARD: check whether X lies within [0,1]
% If not, linearly scale each column to [0,1]
% =========================================================
tol_x = 1e-12;

X_min = min(X, [], 1);           % 1 x p
X_max = max(X, [], 1);           % 1 x p
X_range = X_max - X_min;         % 1 x p

outside_mask = (X_min < -tol_x) | (X_max > 1 + tol_x);
did_scale_X = any(outside_mask);

X_input_original = X;

if did_scale_X
    fprintf('\nWarning: In provided X, some or all of the covariates lie outside the bound [0,1].\n');
    fprintf('X will be linearly scaled within [0,1] before model fitting.\n');
    
    X_scaled = X;
    
    for j = 1:p
        if X_range(j) > tol_x
            X_scaled(:,j) = (X(:,j) - X_min(j)) ./ X_range(j);
        else
            % Constant column: map to zero
            X_scaled(:,j) = 0;
        end
    end
    
    X = X_scaled;
end

% =========================================================
% STAGE 1.1: FULL-SAMPLE UNIVARIATE QR BASIS
% =========================================================
fprintf('\n *** Running UNIQUE, Stage 1.1: Computing univariate coeffs...\n');
[beta0_univ_hat, beta_univ_hat] = ...
    compute_UNIQUE_univariate_basis(X, Y, tau_grid);

% =========================================================
% STAGE 1.2: GROUPED CROSS-FITTED META-FEATURES
% =========================================================
fprintf('\n *** Running UNIQUE, Stage 1.2: Computing out-of-sample cross-fitting coeffs...\n');
[eta_CF, fold_id] = compute_grouped_crossfit_eta( ...
    X, Y, tau_grid, opts.num_folds_for_crossfit, opts.verbose_crossfit);

% =========================================================
% STAGE 2.1: ADAPTIVE WEIGHTS
% =========================================================
fprintf('\n *** Running UNIQUE, Stage 2.1: Computing adaptive weights w_{jk} ...\n');
[beta0_hat_mult, beta_hat_mult, qfit_j, signal_score, omega] = ...
    compute_UNIQUE_adaptive_weights(X, Y, tau_grid, opts.eps_w);


% =========================================================
% STAGE 2.2: FEASIBLE INITIALIZATION
% =========================================================
fprintf('\n *** Running UNIQUE, Stage 2.2: Finding warm start before SAPS ...\n');
init = build_unilasso_feasible_init( ...
    X, Y, tau_grid, signal_score, beta0_univ_hat, beta_univ_hat, ...
    opts.same_sign_mask, opts.t_row, opts.gate_thr, opts.lb, opts.ub, ...
    opts.theta_cap, opts.beta0_contrib_cap, opts.verbose_stage_2_init);

theta0_init = init.theta0_init;
theta_init  = init.theta_init;
s_init      = init.s_init; %#ok<NASGU>

if opts.rand_feasible_initiation == 1
    max_attempts = 100;
    [theta0_init, theta_init, attempts_made] = ...
    generate_feasible_random_theta(p, K, beta0_univ_hat, beta_univ_hat, ...
                                   opts.same_sign_mask, opts.lb, opts.ub, max_attempts);
    fprintf('Random feasible initialization used. Attempts = %d\n', attempts_made);
end

% =========================================================
% STAGE 2.3: Lambda tuning
%   If num_folds_for_final_lambda = 1:
%       one random 80/20 split
%   If num_folds_for_final_lambda >= 2:
%       K-fold CV
% =========================================================
fprintf('\n *** Running UNIQUE, Stage 2.3: Finding optimal lambda before full-data fitting ...\n');

L = numel(opts.lambda_grid);
nfold_lambda = opts.num_folds_for_final_lambda;

theta0_path  = zeros(K, L);
theta_path   = zeros(p, K, L);
f_path       = zeros(L,1);
cv_loss_path = zeros(L,1);

% ---------------------------------------------------------
% Case 1: Single 80/20 split
% ---------------------------------------------------------
if nfold_lambda == 1

    n_train = round(0.8 * n);
    perm_outer = randperm(n);

    train_idx_outer = perm_outer(1:n_train);
    test_idx_outer  = perm_outer(n_train+1:end);

    Y_train = Y(train_idx_outer);
    Y_test  = Y(test_idx_outer);

    eta_CF_train = eta_CF(train_idx_outer,:,:);
    eta_CF_test  = eta_CF(test_idx_outer,:,:);

    if opts.verbose_stage_2_tuning == 1
        fprintf('\n > Single 80/20 split for Stage-2 tuning: n_train = %d, n_test = %d\n', ...
            numel(Y_train), numel(Y_test));
    end

    theta0_ws = theta0_init;
    theta_ws  = theta_init;

    for ell = 1:L
        lambda = opts.lambda_grid(ell);

        if opts.verbose_stage_2_tuning == 1
            fprintf('\n  Fold 1 / 1: Lambda %d / %d  (lambda = %.4g)\n', ell, L, lambda);
        end

        obj_train = @(th) unilasso_objective(th, Y_train, eta_CF_train, tau_grid, omega, lambda);

        [theta0_best_tmp, theta_best_tmp, f_best_tmp, ~] = ...
            SAPS_NCQR(obj_train, theta0_ws, theta_ws, ...
            beta0_univ_hat, beta_univ_hat, ...
            opts.same_sign_mask, opts.lb, opts.ub, opts.params);

        theta0_path(:,ell)  = theta0_best_tmp;
        theta_path(:,:,ell) = theta_best_tmp;
        f_path(ell)         = f_best_tmp;

        cv_loss_path(ell) = unilasso_check_loss(theta0_best_tmp, theta_best_tmp, ...
            Y_test, eta_CF_test, tau_grid);

        if opts.verbose_stage_2_tuning == 1
            fprintf('  Train objective = %.6e\n', f_best_tmp);
            fprintf('  Test check loss = %.6e\n', cv_loss_path(ell));
        end

        theta0_ws = theta0_best_tmp;
        theta_ws  = theta_best_tmp;
    end

% ---------------------------------------------------------
% Case 2: K-fold CV
% ---------------------------------------------------------
else

    if nfold_lambda > n
        error('opts.num_folds_for_final_lambda cannot exceed n.');
    end

    if opts.verbose_stage_2_tuning == 1
        fprintf('\nUsing %d-fold CV for Stage-2 tuning.\n', nfold_lambda);
    end

    % Balanced folds
    base_size = floor(n / nfold_lambda);
    remainder = mod(n, nfold_lambda);

    perm = randperm(n);
    fold_id_lambda = zeros(n,1);

    start_idx = 1;
    for m = 1:nfold_lambda
        if m <= remainder
            fold_size = base_size + 1;
        else
            fold_size = base_size;
        end

        idx = perm(start_idx:start_idx + fold_size - 1);
        fold_id_lambda(idx) = m;
        start_idx = start_idx + fold_size;
    end

    cv_loss_folds = zeros(L, nfold_lambda);

    % Optional: store path from first fold for plotting/warm-start reference
    theta0_path_first = zeros(K, L);
    theta_path_first  = zeros(p, K, L);
    f_path_first      = zeros(L,1);

    for m = 1:nfold_lambda

        test_idx  = (fold_id_lambda == m);
        train_idx = ~test_idx;

        Y_train = Y(train_idx);
        Y_test  = Y(test_idx);

        eta_CF_train = eta_CF(train_idx,:,:);
        eta_CF_test  = eta_CF(test_idx,:,:);

        if opts.verbose_stage_2_tuning == 1
            fprintf('\n > Fold %d / %d: n_train = %d, n_test = %d\n', ...
                m, nfold_lambda, numel(Y_train), numel(Y_test));
        end

        theta0_ws = theta0_init;
        theta_ws  = theta_init;

        for ell = 1:L
            lambda = opts.lambda_grid(ell);

            if opts.verbose_stage_2_tuning == 1
                fprintf('  Fold %d / %d: Lambda %d / %d  (lambda = %.4g)\n',m, nfold_lambda, ell, L, lambda);
            end

            obj_train = @(th) unilasso_objective(th, Y_train, eta_CF_train, tau_grid, omega, lambda);

            [theta0_best_tmp, theta_best_tmp, f_best_tmp, ~] = ...
                SAPS_NCQR(obj_train, theta0_ws, theta_ws, ...
                beta0_univ_hat, beta_univ_hat, ...
                opts.same_sign_mask, opts.lb, opts.ub, opts.params);

            cv_loss_folds(ell, m) = unilasso_check_loss(theta0_best_tmp, theta_best_tmp, ...
                Y_test, eta_CF_test, tau_grid);

            if m == 1
                theta0_path_first(:,ell)  = theta0_best_tmp;
                theta_path_first(:,:,ell) = theta_best_tmp;
                f_path_first(ell)         = f_best_tmp;
            end

            if opts.verbose_stage_2_tuning == 1
                fprintf('    Train objective = %.6e\n', f_best_tmp);
                fprintf('    Validation check loss = %.6e\n', cv_loss_folds(ell, m));
            end

            theta0_ws = theta0_best_tmp;
            theta_ws  = theta_best_tmp;
        end
    end

    cv_loss_path = mean(cv_loss_folds, 2);

    % Keep first-fold path objects for downstream plotting / initialization
    theta0_path = theta0_path_first;
    theta_path  = theta_path_first;
    f_path      = f_path_first;
end

% =========================================================
% SELECT LAMBDA
% =========================================================
[cv_opt, idx_opt] = min(cv_loss_path);

if opts.use_fixed_lambda_opt == 1
    lambda_opt = opts.fixed_lambda_opt;
    [~, idx_opt] = min(abs(opts.lambda_grid - lambda_opt));
    cv_opt = cv_loss_path(idx_opt);
else
    lambda_opt = opts.lambda_grid(idx_opt);
end

if opts.verbose_stage_2_tuning == 1
    fprintf('\nOptimal lambda selected = %.6g\n', lambda_opt);
    fprintf('Optimal CV/check loss   = %.6e\n', cv_opt);
end

% =========================================================
% OPTIONAL PLOT: lambda vs CV check loss
% =========================================================
if opts.plot_cv_path == 1
    
    figure('Color','w');
    
    semilogx(opts.lambda_grid, cv_loss_path, '-o', ...
        'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    
    plot(lambda_opt, cv_loss_path(idx_opt), 'rp', ...
        'MarkerSize', 12, 'MarkerFaceColor', 'r');
    
    xlabel('\lambda', 'FontSize', 13);
    ylabel('Check loss (test)', 'FontSize', 13);
    
    if isfield(opts, 'plot_title_cv') && ~isempty(opts.plot_title_cv)
        title(opts.plot_title_cv, 'FontSize', 13);
    else
        title('Lambda vs CV Check Loss', 'FontSize', 13);
    end
    
    grid on;
    box on;
    set(gca, 'FontSize', 12);
    
    legend({'CV loss', 'Selected \lambda'}, 'Location', 'best');
end

% =========================================================
% STAGE 3: FINAL FULL-DATA REFIT
% =========================================================

fprintf('\n *** Running UNIQUE, Stage 3: Full data fitting ...\n');

if opts.verbose_stage_3 == 1
    fprintf('\nActivating SAPS_NCQR on FULL data at lambda_opt = %.6g ...\n', lambda_opt);
end

obj_full = @(th) unilasso_objective(th, Y, eta_CF, tau_grid, omega, lambda_opt);

theta0_start_full = theta0_path(:, idx_opt);
theta_start_full  = theta_path(:,:,idx_opt);

[theta0_best, theta_best, f_best_full, ~] = ...
    SAPS_NCQR(obj_full, theta0_start_full, theta_start_full, ...
    beta0_univ_hat, beta_univ_hat, ...
    opts.same_sign_mask, opts.lb, opts.ub, opts.params);

if opts.verbose_stage_3 == 1
    fprintf('Final full-data objective value = %.6e\n', f_best_full);
end

% j = 30;
% 
% fprintf('\n================ DEBUG ROW %d ================\n', j);
% 
% fprintf('beta_univ_hat row %d:\n', j);
% disp(beta_univ_hat(j,:));
% 
% fprintf('beta0_univ_hat row %d:\n', j);
% disp(beta0_univ_hat(j,:));
% 
% fprintf('theta_init_raw row %d:\n', j);
% disp(init.theta_init_raw(j,:));
% 
% fprintf('theta_init row %d:\n', j);
% disp(init.theta_init(j,:));
% 
% fprintf('theta_best row %d:\n', j);
% disp(theta_best(j,:));
% 
% fprintf('deployed beta row %d:\n', j);
% disp(theta_best(j,:) .* beta_univ_hat(j,:));


% =========================================================
% DEPLOYED COEFFICIENTS
% =========================================================
beta0_hat_unilasso = zeros(K,1);
beta_hat_unilasso  = zeros(p,K);

for k = 1:K
    beta0_hat_unilasso(k)  = theta0_best(k) + sum(theta_best(:,k) .* beta0_univ_hat(:,k));
    beta_hat_unilasso(:,k) = theta_best(:,k) .* beta_univ_hat(:,k);
end

% fprintf('\n=========== FINAL INTERCEPT DECOMPOSITION ===========\n');
% 
% disp('theta0_best:');
% disp(theta0_best(:)');
% 
% disp('beta0_hat_unilasso:');
% disp(beta0_hat_unilasso(:)');
% 
% contrib_beta0 = theta_best .* beta0_univ_hat;   % p x K
% 
% disp('sum_j theta_best(j,k)*beta0_univ_hat(j,k):');
% disp(sum(contrib_beta0,1));
% 
% disp('check theta0_best + above sum:');
% disp((theta0_best(:)' + sum(contrib_beta0,1)));
% 
% k_list = [8 9];
% 
% for kk = k_list
%     fprintf('\nTop intercept contributors at tau = %.2f\n', tau_grid(kk));
%     contrib_k = theta_best(:,kk) .* beta0_univ_hat(:,kk);
%     [vals, ord] = sort(abs(contrib_k), 'descend');
%     disp(table(ord(1:10), contrib_k(ord(1:10)), ...
%         'VariableNames', {'row','theta_beta0_contrib'}));
% end
% 
% fprintf('\nFinal deployed beta0_hat_unilasso:\n');
% disp(beta0_hat_unilasso(:)');
% 
% fprintf('\nFinal deployed beta_hat_unilasso rows 1:5 and 30:\n');
% disp(beta_hat_unilasso([1 2 3 4 30], :));
% =========================================================
% Back-transform UNIQUE coefficients to original X scale
% =========================================================

beta0_hat_unilasso_raw = beta0_hat_unilasso;
beta_hat_unilasso_raw = beta_hat_unilasso;

if did_scale_X
    beta0_hat_unilasso_scaled = beta0_hat_unilasso;
    beta_hat_unilasso_scaled  = beta_hat_unilasso;

    beta0_hat_unilasso_orig = zeros(K,1);
    beta_hat_unilasso_orig  = zeros(p,K);

    inv_range = zeros(1,p);
    nz = X_range > tol_x;
    inv_range(nz) = 1 ./ X_range(nz);

    for k = 1:K
        beta_hat_unilasso_orig(:,k) = beta_hat_unilasso_scaled(:,k) .* inv_range(:);

        beta0_hat_unilasso_orig(k) = beta0_hat_unilasso_scaled(k) ...
            - sum(beta_hat_unilasso_scaled(:,k)' .* (X_min .* inv_range));
    end

    beta0_hat_unilasso = beta0_hat_unilasso_orig;
    beta_hat_unilasso  = beta_hat_unilasso_orig;

    fprintf('\nEstimated coefficients and Qhat are adjusted in original scale.\n');
end

Qhat = @(x) ones(size(x,1),1) * beta0_hat_unilasso(:)' + x * beta_hat_unilasso;

q_pred_all = Qhat(X_input_original);
min_gap = min(min(diff(q_pred_all,1,2)));
if opts.verbose_stage_3 == 1
    fprintf('Sanity check on observed X: min adjacent gap = %.3e\n', min_gap);
end
time_unilasso = toc;

fprintf('\n DONE..."We are UNIQUE, and we respect others’ boundaries" \n');
fprintf('\n UNIQUE computation time: = %.1f\n', time_unilasso);
% =========================================================
% STORE RESULTS
% =========================================================
results = struct();
results.time_unilasso = time_unilasso;
results.X = X;
results.Y = Y;
results.tau_grid = tau_grid;
results.opts = opts;

results.fold_id = fold_id;
results.eta_CF = eta_CF;

results.beta0_univ_hat = beta0_univ_hat;
results.beta_univ_hat  = beta_univ_hat;

results.beta0_hat_mult = beta0_hat_mult;
results.beta_hat_mult  = beta_hat_mult;
results.qfit_j         = qfit_j;
results.signal_score   = signal_score;
results.omega          = omega;

results.init = init;
results.theta0_init = theta0_init;
results.theta_init  = theta_init;

results.theta0_path = theta0_path;
results.theta_path  = theta_path;
results.f_path      = f_path;
results.cv_loss_path = cv_loss_path;

results.lambda_opt = lambda_opt;
results.idx_opt    = idx_opt;
results.cv_opt     = cv_opt;

results.theta0_best = theta0_best;
results.theta_best  = theta_best;
results.f_best_full = f_best_full;

results.beta0_hat_unilasso = beta0_hat_unilasso;
results.beta_hat_unilasso  = beta_hat_unilasso;
results.beta0_hat_unilasso_raw = beta0_hat_unilasso_raw;
results.beta_hat_unilasso_raw  = beta_hat_unilasso_raw;

results.Qhat = Qhat;
results.min_gap = min_gap;

results.X_original = X_input_original;
results.did_scale_X = did_scale_X;
results.X_min = X_min;
results.X_max = X_max;
results.X_range = X_range;
results.X_used_for_fitting = X;
    

% =========================================================
% QR-LASSO (Optional)
% =========================================================
if opts.run_qrlasso == 1
    fprintf('\n *** Running QR Lasso...\n');
    tic;
    [beta0_qrlasso, beta_qrlasso, lambda_opt_qrlasso] = ...
        qr_lasso_cv(X, Y, tau_grid, opts.lambda_grid_QRLASSO, opts.num_folds_qrlasso);
    
    % =========================================================
    % Back-transform QR-LASSO coefficients to original X scale
    % =========================================================
    if did_scale_X
        beta0_qrlasso_scaled = beta0_qrlasso;
        beta_qrlasso_scaled  = beta_qrlasso;
        
        if isrow(beta0_qrlasso_scaled)
            beta0_qrlasso_scaled = beta0_qrlasso_scaled(:);
        end
        
        beta0_qrlasso_orig = zeros(K,1);
        beta_qrlasso_orig  = zeros(p,K);
        
        inv_range = zeros(1,p);
        nz = X_range > tol_x;
        inv_range(nz) = 1 ./ X_range(nz);
        
        for k = 1:K
            beta_qrlasso_orig(:,k) = beta_qrlasso_scaled(:,k) .* inv_range(:);
            
            beta0_qrlasso_orig(k) = beta0_qrlasso_scaled(k) ...
                - sum(beta_qrlasso_scaled(:,k)' .* (X_min .* inv_range));
        end
        
        beta0_qrlasso = beta0_qrlasso_orig;
        beta_qrlasso  = beta_qrlasso_orig;
    end
    % =========================================================
    time_qrlasso = toc;
    results.time_qrlasso        = time_qrlasso;
    results.beta0_qrlasso       = beta0_qrlasso;
    results.beta_qrlasso        = beta_qrlasso;
    results.lambda_opt_qrlasso  = lambda_opt_qrlasso;
    fprintf('\n DONE...\n');
    fprintf('\nQRLasso computation time: = %.1f\n', time_qrlasso);
end


% Optional truth-based evaluation
if isfield(opts, 'beta0_true_grid') && isfield(opts, 'beta_true_grid')
    results.metrics = evaluation_post_UNIQUE(results, opts);
end
end