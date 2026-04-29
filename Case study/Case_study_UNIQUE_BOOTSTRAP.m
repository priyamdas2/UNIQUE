clear; clc;
addpath('./UNIQUE source funs/');
addpath('./SAPS/');

datadir_real = 'PPMI data';
bootdir = 'Bootstrap for CI';
if ~exist(bootdir, 'dir')
    mkdir(bootdir);
end

%% =======================
%  USER OPTION
%% =======================
use_dataset = 2;         % must be 1 / 2 / 3, 2 = main analysis data
Num_bootstrap = 1000;
strat_from = 501;
if ~ismember(use_dataset, [1, 2, 3])
    error('use_dataset must be either 1, 2 or 3.');
end

%% =======================
%  DATA LOADING (REAL DATA)
%% =======================

% Quantile grid
tau_grid = 0.1:0.05:0.9;
tau_grid = tau_grid(:)';   % ensure row vector
K = length(tau_grid);

% File names
fname_X = fullfile(datadir_real, sprintf('X_set_%d_final.csv', use_dataset));
fname_Y = fullfile(datadir_real, sprintf('Y_set_%d.csv', use_dataset));

% ---------------------------------------------------------
% Load X with variable names preserved
% ---------------------------------------------------------
TX = readtable(fname_X, 'VariableNamingRule', 'preserve');

if width(TX) >= 2
    first_col = TX{:,1};
    if isnumeric(first_col) && isequal(first_col(:), (1:height(TX))')
        TX(:,1) = [];
    end
end

x_var_names = string(TX.Properties.VariableNames);
X = table2array(TX);

% ---------------------------------------------------------
% Load Y
% ---------------------------------------------------------
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

% ---------------------------------------------------------
% Basic safety checks / alignment
% ---------------------------------------------------------
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
%  SELECTIVE SIGN MASK
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

%% =======================
%  UNIQUE + SAPS PARAMETERS
%% =======================

% UNIQUE tuning
t_row = 0.01;     % more relaxed for Bootstrap
gate_thr = 0.01;  % more relaxed for Bootstrap
num_folds_for_crossfit = 10;
num_folds_for_final_lambda = 5;
lambda_grid_UNIQUE = sort(logspace(-5, 5, 20));

% UNIQUE warm start tuning
rand_feasible_initiation = 0;
theta_cap                = 20;
beta0_contrib_cap        = 20;

%% =======================
%  UNIQUE OPTIONS
%% =======================

opts = struct();

% IMPORTANT: do NOT run QR-LASSO
opts.run_qrlasso = 0;

% Silence all printing controlled through opts
opts.verbose_crossfit       = 1;
opts.verbose_stage_2_init   = 1;
opts.verbose_stage_2_tuning = 1;
opts.verbose_stage_3        = 1;

% UNIQUE parameters
opts.same_sign_mask             = same_sign_mask;
opts.num_folds_for_crossfit     = num_folds_for_crossfit;
opts.num_folds_for_final_lambda = num_folds_for_final_lambda;
opts.lambda_grid                = lambda_grid_UNIQUE;

% warm starting point tuning
opts.rand_feasible_initiation = rand_feasible_initiation;
opts.t_row                    = t_row;
opts.gate_thr                 = gate_thr;
opts.theta_cap                = theta_cap;
opts.beta0_contrib_cap        = beta0_contrib_cap;

% SAPS tuning
opts.domain_magnitude = 100;

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
opts.params          = params;

%% =======================
%  START PARALLEL POOL IF NEEDED
%% =======================
poolobj = gcp('nocreate');
if isempty(poolobj)
    parpool;
end

%% =======================
%  BOOTSTRAP IN PARALLEL
%% =======================
parfor rep = strat_from:Num_bootstrap

    rng(rep, 'twister');

    success = false;
    err_msg = '';
    boot_idx = [];
    coef_orig_unique = [];
    coef_scaled_unique = [];

    try
        % SRSWR bootstrap indices
        boot_idx = randsample(n, n, true);

        % Bootstrap sample
        Xb = X(boot_idx, :);
        Yb = Y(boot_idx, :);

        % Fit UNIQUE
        results = UNIQUE(Xb, Yb, tau_grid, opts);

        % Extract original-scale coefficients
        if ~isfield(results, 'beta0_hat_unilasso')
            error('results.beta0_hat_unilasso not found.');
        end
        if ~isfield(results, 'beta_hat_unilasso')
            error('results.beta_hat_unilasso not found.');
        end
        if ~isfield(results, 'X_min') || ~isfield(results, 'X_range')
            error('results.X_min / results.X_range not found.');
        end

        beta0_orig_unique = results.beta0_hat_unilasso(:);   % K x 1
        beta_orig_unique  = results.beta_hat_unilasso;       % p x K

        if ~isequal(size(beta_orig_unique), [p, K])
            error('results.beta_hat_unilasso must be of size p x K.');
        end
        if length(beta0_orig_unique) ~= K
            error('results.beta0_hat_unilasso must have length K.');
        end

        % Reconstruct internal scaled-[0,1] coefficients
        X_min   = results.X_min;
        X_range = results.X_range;

        beta_scaled_unique  = zeros(p, K);
        beta0_scaled_unique = zeros(K, 1);

        for k = 1:K
            beta_scaled_unique(:, k) = beta_orig_unique(:, k) .* X_range(:);
            beta0_scaled_unique(k)   = beta0_orig_unique(k) + sum(beta_orig_unique(:, k)' .* X_min);
        end

        % Append intercept row
        coef_orig_unique   = [beta0_orig_unique'; beta_orig_unique];
        coef_scaled_unique = [beta0_scaled_unique'; beta_scaled_unique];

        success = true;

    catch ME
        err_msg = getReport(ME, 'basic', 'hyperlinks', 'off');
    end
    
    outfile_orig = fullfile(bootdir, ...
        sprintf('UNIQUE_boot_rep_%02d_set_%d_orig.csv', rep, use_dataset));
    
    outfile_scaled = fullfile(bootdir, ...
        sprintf('UNIQUE_boot_rep_%02d_set_%d_scaled.csv', rep, use_dataset));
    
    writematrix(coef_orig_unique, outfile_orig);
    writematrix(coef_scaled_unique, outfile_scaled);
end