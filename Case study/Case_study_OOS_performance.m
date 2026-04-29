clear; clc;

addpath('./UNIQUE source funs/');
addpath('./SAPS/');
addpath('./BONDELL source funs/');

%% =======================
%  OUTPUT DIRECTORY
%% =======================
outdir = 'Output';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

%% =======================
%  USER OPTIONS
%% =======================
rng(2026);              % one-time split seed
use_dataset = 2;        % must be 1 / 2 / 3
train_frac = 0.80;
sel_thr = 0.1;          % selection threshold for fitted coefficient entries

if ~ismember(use_dataset, [1, 2, 3])
    error('use_dataset must be either 1, 2 or 3.');
end

%% =======================
%  DATA LOADING
%% =======================
datadir_real = 'PPMI data';

tau_grid = 0.1:0.05:0.9;
tau_grid = tau_grid(:)';
K = length(tau_grid);

fname_X = fullfile(datadir_real, sprintf('X_set_%d_final.csv', use_dataset));
fname_Y = fullfile(datadir_real, sprintf('Y_set_%d.csv', use_dataset));

TX = readtable(fname_X, 'VariableNamingRule', 'preserve');
if width(TX) >= 2
    first_col = TX{:,1};
    if isnumeric(first_col) && isequal(first_col(:), (1:height(TX))')
        TX(:,1) = [];
    end
end
x_var_names = string(TX.Properties.VariableNames);
X = table2array(TX);

TY = readtable(fname_Y, 'VariableNamingRule', 'preserve');
if width(TY) >= 2
    first_col_y = TY{:,1};
    if isnumeric(first_col_y) && isequal(first_col_y(:), (1:height(TY))')
        TY(:,1) = [];
    end
end
Y = table2array(TY);

if isrow(Y)
    Y = Y(:);
end

if ndims(X) ~= 2
    error('Loaded X is not a 2D matrix.');
end

if size(Y,2) ~= 1
    if size(Y,1) == 1
        Y = Y';
    else
        error('Loaded Y must be a vector / single-column matrix.');
    end
end

if size(X,1) ~= length(Y)
    if size(X,2) == length(Y)
        X = X';
    else
        error('X and Y have incompatible dimensions.');
    end
end

[n, p] = size(X);

if length(x_var_names) ~= p
    error('Number of X variable names does not match number of X columns.');
end

%% =======================
%  TRAIN / TEST SPLIT
%% =======================
perm = randperm(n);
n_train = floor(train_frac * n);
idx_train = perm(1:n_train);
idx_test  = perm(n_train+1:end);

X_train = X(idx_train, :);
Y_train = Y(idx_train, :);

X_test  = X(idx_test, :);
Y_test  = Y(idx_test, :);

%% =======================
%  CONSTRAINED VARIABLES
%% =======================
same_sign_mask = false(p, K);

switch use_dataset
    case 1
        constrained_vars = [
            "hvlt_immediaterecall","HVLTRDLY","hvlt_discrimination","HVLTFPRL", ...
            "HVLTREC","hvlt_retention","VLTANIM","SDMTOTAL","bjlot","lns", ...
            "gds","upsit","pigd","rem","scopa", ...
            "updrs2_score","updrs3_score","MCI_testscores","age"
            ];

    case 2
        constrained_vars = [
            "hvlt_immediaterecall","HVLTRDLY","hvlt_discrimination","HVLTFPRL", ...
            "HVLTREC","hvlt_retention","VLTANIM","SDMTOTAL","bjlot","lns", ...
            "gds","upsit","pigd","rem","scopa", ...
            "updrs2_score","updrs3_score","MCI_testscores","age"
            ];

    case 3
        constrained_vars = [
            "age","gds","upsit","pigd","rem","scopa", ...
            "updrs2_score","updrs3_score"
            ];

    otherwise
        error('Unsupported use_dataset.');
end

for j = 1:p
    if any(strcmp(x_var_names(j), constrained_vars))
        same_sign_mask(j, :) = true;
    end
end

constrained_row_mask = any(same_sign_mask, 2);

%% =======================
%  TRAINING MARGINAL SIGN MATRIX (p x K)
%% =======================
marg_beta_mat = zeros(p, K);

for j = 1:p
    xj = X_train(:, j);
    for k = 1:K
        tau = tau_grid(k);
        [~, bj] = univariate_quantile_regression(xj, Y_train, tau);
        marg_beta_mat(j, k) = bj;
    end
end

% No threshold for marginal sign determination
marg_sign_mat = sign(marg_beta_mat);

%% =======================
%  METRIC STORAGE
%% =======================
tau50_idx = find(abs(tau_grid - 0.5) < 1e-12, 1);
if isempty(tau50_idx)
    error('tau = 0.5 not found in tau_grid.');
end

method_names = {'UNIQUE'; 'QRLASSO'; 'BONDELL'};
RMSE = NaN(3,1);
MAE  = NaN(3,1);
MeanPinballLoss = NaN(3,1);
PropSignMatch = NaN(3,1);
PropSignReversed = NaN(3,1);
NumSelected = NaN(3,1);
CompTime = NaN(3,1);

%% =======================
%  UNIQUE + QRLASSO OPTIONS
%% =======================
t_row = 0.05;   % default 0.05
gate_thr = 0.05; % default 0.05
num_folds_for_crossfit = 10;
num_folds_for_final_lambda = 5;
lambda_grid_UNIQUE = sort(logspace(-5, 5, 20));

rand_feasible_initiation = 0;
theta_cap                = 50;
beta0_contrib_cap        = 50;

num_folds_qrlasso   = 5;
lambda_grid_QRLASSO = sort(logspace(-5, 5, 20));

opts_unique = struct();
opts_unique.run_qrlasso = 1;
opts_unique.num_folds_qrlasso   = num_folds_qrlasso;
opts_unique.lambda_grid_QRLASSO = lambda_grid_QRLASSO;

% Keep silent for OOS run
opts_unique.verbose_crossfit       = 1;
opts_unique.verbose_stage_2_init   = 1;
opts_unique.verbose_stage_2_tuning = 1;
opts_unique.verbose_stage_3        = 1;

opts_unique.same_sign_mask             = same_sign_mask;
opts_unique.num_folds_for_crossfit     = num_folds_for_crossfit;
opts_unique.num_folds_for_final_lambda = num_folds_for_final_lambda;
opts_unique.lambda_grid                = lambda_grid_UNIQUE;

opts_unique.rand_feasible_initiation = rand_feasible_initiation;
opts_unique.t_row                    = t_row;
opts_unique.gate_thr                 = gate_thr;
opts_unique.theta_cap                = theta_cap;
opts_unique.beta0_contrib_cap        = beta0_contrib_cap;

opts_unique.domain_magnitude = 100;

params               = struct();
params.s_init        = 0.02;
params.s_inc         = 2;
params.s_dec         = 2;
params.p_inc         = 2;
params.p_dec         = 2;
params.m             = 500;
params.n_hit_and_run = 5;
params.T             = round(3000 * log(length(tau_grid) + p*length(tau_grid)));
params.M             = 10 * (length(tau_grid) + p*length(tau_grid));
params.epsilon       = 1e-12;
params.feas_tol      = 1e-8;
params.c_log         = 0.001 * log(length(tau_grid) + p*length(tau_grid));
opts_unique.params   = params;

%% =======================
%  FIT UNIQUE (+ internal QRLASSO)
%% =======================
results_unique = UNIQUE(X_train, Y_train, tau_grid, opts_unique);

% Exact times from output
CompTime(1) = results_unique.time_unilasso;
CompTime(2) = results_unique.time_qrlasso;

% ---------------------------------------
% UNIQUE: original-scale coefficients
% used for prediction on original X_test
% ---------------------------------------
beta0_unique_orig = results_unique.beta0_hat_unilasso(:);
beta_unique_orig  = results_unique.beta_hat_unilasso;

Qhat_test_unique = predict_quantile_matrix(X_test, beta0_unique_orig, beta_unique_orig);
yhat_test_unique = Qhat_test_unique(:, tau50_idx);

RMSE(1) = sqrt(mean((Y_test - yhat_test_unique).^2));
MAE(1)  = mean(abs(Y_test - yhat_test_unique));
MeanPinballLoss(1) = mean_pinball_loss(Y_test, Qhat_test_unique, tau_grid);

% ---------------------------------------
% UNIQUE: internal/raw [0,1]-scale beta
% used for sign determination
% ---------------------------------------

beta_unique_raw = results_unique.beta_hat_unilasso_raw;
NumSelected(1) = count_selected_variables(beta_unique_raw, sel_thr);

[PropSignMatch(1), PropSignReversed(1)] = ...
    sign_metrics_from_matrix(beta_unique_raw, marg_sign_mat, constrained_row_mask, sel_thr);

% ---------------------------------------
% QRLASSO: original-scale coefficients
% used for prediction on original X_test
% ---------------------------------------
beta0_qrlasso_orig = results_unique.beta0_qrlasso(:);
beta_qrlasso_orig  = results_unique.beta_qrlasso;

Qhat_test_qrlasso = predict_quantile_matrix(X_test, beta0_qrlasso_orig, beta_qrlasso_orig);
yhat_test_qrlasso = Qhat_test_qrlasso(:, tau50_idx);

RMSE(2) = sqrt(mean((Y_test - yhat_test_qrlasso).^2));
MAE(2)  = mean(abs(Y_test - yhat_test_qrlasso));
MeanPinballLoss(2) = mean_pinball_loss(Y_test, Qhat_test_qrlasso, tau_grid);

% ---------------------------------------
% QRLASSO: internal/raw [0,1]-scale beta
% used for sign determination
% ---------------------------------------
beta_qrlasso_raw = reconstruct_internal_beta( ...
    results_unique.beta_qrlasso, ...
    results_unique.X_range, ...
    results_unique.did_scale_X);
NumSelected(2) = count_selected_variables(beta_qrlasso_raw, sel_thr);

[PropSignMatch(2), PropSignReversed(2)] = ...
    sign_metrics_from_matrix(beta_qrlasso_raw, marg_sign_mat, constrained_row_mask, sel_thr);

%% =======================
%  FIT BONDELL
%% =======================
opts_bondell = struct();
opts_bondell.verbose = 0;
opts_bondell.scale_X_to_unit = 1;
opts_bondell.linprog_display = 'none';

results_bondell = BONDLL_NCQR(X_train, Y_train, tau_grid, opts_bondell);

% Exact time from output
CompTime(3) = results_bondell.time_bondell;

% ---------------------------------------
% BONDELL: original-scale coefficients
% used for prediction on original X_test
% ---------------------------------------
beta0_bondell_orig = results_bondell.beta0_hat_bondell(:);
beta_bondell_orig  = results_bondell.beta_hat_bondell;

Qhat_test_bondell = predict_quantile_matrix(X_test, beta0_bondell_orig, beta_bondell_orig);
yhat_test_bondell = Qhat_test_bondell(:, tau50_idx);

RMSE(3) = sqrt(mean((Y_test - yhat_test_bondell).^2));
MAE(3)  = mean(abs(Y_test - yhat_test_bondell));
MeanPinballLoss(3) = mean_pinball_loss(Y_test, Qhat_test_bondell, tau_grid);

% ---------------------------------------
% BONDELL: internal/raw [0,1]-scale beta
% used for sign determination
% ---------------------------------------
beta_bondell_raw = reconstruct_internal_beta( ...
    results_bondell.beta_hat_bondell, ...
    results_bondell.X_range, ...
    results_bondell.did_scale_X);
NumSelected(3) = count_selected_variables(beta_bondell_raw, sel_thr);

[PropSignMatch(3), PropSignReversed(3)] = ...
    sign_metrics_from_matrix(beta_bondell_raw, marg_sign_mat, constrained_row_mask, sel_thr);

%% =======================
%  SAVE FINAL OOS TABLE
%% =======================
T = table( ...
    method_names, ...
    RMSE, ...
    MAE, ...
    MeanPinballLoss, ...
    PropSignMatch, ...
    PropSignReversed, ...
    NumSelected, ...
    CompTime, ...
    'VariableNames', { ...
        'Method', ...
        'RMSE', ...
        'MAE', ...
        'MeanPinballLoss', ...
        'PropSignMatch', ...
        'PropSignReversed', ...
        'NumSelected', ...
        'CompTime'});

outfile = fullfile(outdir, sprintf('OOS_performance_set_%d.csv', use_dataset));
writetable(T, outfile);

%% =======================
%  LOCAL FUNCTIONS
%% =======================

function Qhat = predict_quantile_matrix(X, beta0, beta)
    % X: n x p
    % beta0: K x 1
    % beta: p x K
    Qhat = X * beta + repmat(beta0(:)', size(X,1), 1);
end

function loss = mean_pinball_loss(Y, Qhat, tau_grid)
    [n, K] = size(Qhat);

    if length(Y) ~= n
        error('Y length mismatch in mean_pinball_loss.');
    end
    if length(tau_grid) ~= K
        error('tau_grid length mismatch in mean_pinball_loss.');
    end

    L = zeros(n, K);
    for k = 1:K
        r = Y - Qhat(:,k);
        tau = tau_grid(k);
        L(:,k) = r .* (tau - (r < 0));
    end
    loss = mean(L(:));
end

function beta_raw = reconstruct_internal_beta(beta_orig, X_range, did_scale_X)
    % Reconstruct beta on the internal/raw [0,1]-scaled X scale.
    %
    % If fitting scaled X into [0,1], then
    %   beta_raw = beta_orig .* X_range
    %
    % If no scaling was applied internally, beta_raw = beta_orig.

    if did_scale_X
        beta_raw = beta_orig .* X_range(:);
    else
        beta_raw = beta_orig;
    end
end

function [prop_match, prop_reversed] = sign_metrics_from_matrix(beta_hat_raw, marg_sign_mat, constrained_row_mask, sel_thr)
    % beta_hat_raw: p x K, on internal/raw [0,1]-X scale
    % marg_sign_mat: p x K, training marginal sign matrix
    % constrained_row_mask: p x 1 logical
    % sel_thr: selection threshold on |beta_hat_raw|

    constrained_mask_full = repmat(constrained_row_mask(:), 1, size(beta_hat_raw,2));

    % Method-specific selected set among constrained entries only
    selected_mask = constrained_mask_full & (abs(beta_hat_raw) > sel_thr);

    denom = sum(selected_mask(:));

    if denom == 0
        prop_match = NaN;
        prop_reversed = NaN;
        return;
    end

    fitted_sign_mat = sign(beta_hat_raw);

    prop_match = sum(fitted_sign_mat(selected_mask) == marg_sign_mat(selected_mask)) / denom;
    prop_reversed = sum(fitted_sign_mat(selected_mask) == -marg_sign_mat(selected_mask)) / denom;
end

function [b0, b1] = univariate_quantile_regression(x, y, tau)
    % Univariate quantile regression via LP:
    %   min_{b0,b1} sum rho_tau(y - b0 - b1*x)

    x = x(:);
    y = y(:);
    n = length(y);

    if length(x) ~= n
        error('x and y length mismatch in univariate_quantile_regression.');
    end

    % z = [b0; b1; u(1:n); v(1:n)]
    f = [0; 0; tau*ones(n,1); (1-tau)*ones(n,1)];

    Aeq = [ones(n,1), x, eye(n), -eye(n)];
    beq = y;

    lb = [-Inf; -Inf; zeros(2*n,1)];
    ub = [];

    opts_lp = optimoptions('linprog', 'Display', 'none');
    z = linprog(f, [], [], Aeq, beq, lb, ub, opts_lp);

    b0 = z(1);
    b1 = z(2);
end

function nsel = count_selected_variables(beta_hat_raw, sel_thr)
    mean_abs_beta = mean(abs(beta_hat_raw), 2);   % p x 1
    nsel = sum(mean_abs_beta > sel_thr);
end