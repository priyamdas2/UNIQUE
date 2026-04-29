% ============================================================
% UNIQUE DEMO
% ============================================================
clear; clc;

addpath('./UNIQUE source funs/');
addpath('./SAPS/');

rng(1);

%% =======================
%  User options
%% =======================

data_file = "YX_data.csv";

% Full fitting grid, same as earlier MATLAB case-study code
tau_grid = 0.1:0.05:0.9;
tau_grid = tau_grid(:)'; 
K = length(tau_grid);


% List of X varaibles to be sign constrained

constrained_vars = ["X01","X02","X03","X05","X07","X11","X13","X17",...
    "X19","X23","X29","X31"];


%% =======================
%  Read processed dataset
%% =======================

T = readtable(data_file, 'VariableNamingRule', 'preserve');

Y = T.Y;
X_table = removevars(T, "Y");

x_var_names = string(X_table.Properties.VariableNames);
X = table2array(X_table);

if isrow(Y)
    Y = Y(:);
end

[n, p] = size(X);

fprintf('--------------------------------------------\n');
fprintf('Preparing to fit UNIQUE\n');
fprintf('n = %d, p = %d, K = %d quantiles\n', n, p, K);
fprintf('--------------------------------------------\n');

%% =======================
%  Selective sign mask for UNIQUE
%% =======================

same_sign_mask = false(p, K);

for j = 1:p
    if any(strcmp(x_var_names(j), constrained_vars))
        same_sign_mask(j, :) = true;
    end
end

fprintf('\nSelective sign mask activated for %d of %d predictors.\n', ...
    sum(any(same_sign_mask, 2)), p);

%% =======================
%  UNIQUE options 
%% =======================

t_row = 0.05; % between 0.02 (generous selection) to 0.1 (strict selection)
gate_thr = 0.05;
num_folds_for_crossfit = 10;
num_folds_for_final_lambda = 5;
lambda_grid_UNIQUE = sort(logspace(-5, 5, 20));

theta_cap                = 50;
beta0_contrib_cap        = 50;

opts_unique = struct();

% No need to run internal QR-LASSO here
opts_unique.run_qrlasso = 0;

opts_unique.verbose_crossfit       = 1; % set 0 to skip printing updates
opts_unique.verbose_stage_2_init   = 1; % set 0 to skip printing updates
opts_unique.verbose_stage_2_tuning = 1; % set 0 to skip printing updates
opts_unique.verbose_stage_3        = 1; % set 0 to skip printing updates

opts_unique.same_sign_mask             = same_sign_mask;
opts_unique.num_folds_for_crossfit     = num_folds_for_crossfit;
opts_unique.num_folds_for_final_lambda = num_folds_for_final_lambda;
opts_unique.lambda_grid                = lambda_grid_UNIQUE;

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

opts_unique.params = params;

%% =======================
%  Fit UNIQUE on full data
%% =======================

fprintf('\n=== Fitting UNIQUE on full processed data ===\n');

results_unique = UNIQUE(X, Y, tau_grid, opts_unique);

beta_unique = results_unique.beta_hat_unilasso;   % p x K, original X scale
beta0_unique = results_unique.beta0_hat_unilasso;   % K x 1, original X scale

% To extract other result metrics, look into all listed returned objects in
% 'results_unique'.
