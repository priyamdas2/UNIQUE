% ============================================================
% NCQR (Bondell) vs UNIQUE coefficient table
% Processed PPMI data, selected variables, selected tau levels
% ============================================================

clear; clc;

addpath('./UNIQUE source funs/');
addpath('./SAPS/');
addpath('./BONDELL source funs/');

rng(1);

%% =======================
%  User options
%% =======================

data_file = "Processed_dataset.csv";

% Full fitting grid, same as earlier MATLAB case-study code
tau_grid = 0.1:0.05:0.9;
tau_grid = tau_grid(:)'; 
K = length(tau_grid);

% Only these tau levels are saved in final coefficient table
taus_to_be_saved = [0.25, 0.5, 0.75];

selected_vars = [
    "hvlt_immediaterecall","HVLTRDLY","hvlt_discrimination","HVLTFPRL", ...
    "HVLTREC","hvlt_retention","VLTANIM","SDMTOTAL","bjlot","lns", ...
    "gds","upsit","pigd","rem","scopa", ...
    "updrs2_score","updrs3_score","age"
];

constrained_vars = selected_vars;

out_file = "Output/NCQR_vs_UNIQUE_all_tau_save_selected_vars.csv";

%% =======================
%  Read processed dataset
%% =======================

T = readtable(data_file, 'VariableNamingRule', 'preserve');

if ~ismember("Y", string(T.Properties.VariableNames))
    error('Processed_dataset.csv must contain outcome column named Y.');
end

Y = T.Y;
X_table = removevars(T, "Y");

x_var_names = string(X_table.Properties.VariableNames);
X = table2array(X_table);

if isrow(Y)
    Y = Y(:);
end

[n, p] = size(X);

fprintf('--------------------------------------------\n');
fprintf('NCQR vs UNIQUE coefficient table\n');
fprintf('n = %d, p = %d, K = %d quantiles\n', n, p, K);
fprintf('--------------------------------------------\n');

%% =======================
%  Check selected variables
%% =======================

selected_available = selected_vars(ismember(selected_vars, x_var_names));
selected_missing   = selected_vars(~ismember(selected_vars, x_var_names));

if ~isempty(selected_missing)
    warning('Selected variables missing from processed dataset: %s', ...
        strjoin(selected_missing, ', '));
end

save_idx = find(ismember(x_var_names, selected_available));

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

t_row = 0.05;
gate_thr = 0.05;
num_folds_for_crossfit = 10;
num_folds_for_final_lambda = 5;
lambda_grid_UNIQUE = sort(logspace(-5, 5, 20));

rand_feasible_initiation = 0;
theta_cap                = 50;
beta0_contrib_cap        = 50;

opts_unique = struct();

% No need to run internal QR-LASSO here
opts_unique.run_qrlasso = 0;

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

opts_unique.params = params;

%% =======================
%  Fit UNIQUE on full data
%% =======================

fprintf('\n=== Fitting UNIQUE on full processed data ===\n');

results_unique = UNIQUE(X, Y, tau_grid, opts_unique);

beta_unique = results_unique.beta_hat_unilasso;   % p x K, original X scale

%% =======================
%  Fit NCQR / Bondell on full data
%% =======================

fprintf('\n=== Fitting NCQR / Bondell on full processed data ===\n');

opts_bondell = struct();
opts_bondell.verbose = 0;
opts_bondell.scale_X_to_unit = 1;
opts_bondell.linprog_display = 'none';

results_bondell = BONDLL_NCQR(X, Y, tau_grid, opts_bondell);

beta_bondell = results_bondell.beta_hat_bondell;  % p x K, original X scale

%% =======================
%  Save coefficient table
%% =======================

tau_save_idx = zeros(size(taus_to_be_saved));

for a = 1:length(taus_to_be_saved)
    [mindiff, idx] = min(abs(tau_grid - taus_to_be_saved(a)));
    if mindiff > 1e-12
        error('Requested tau %.4f is not present in tau_grid.', taus_to_be_saved(a));
    end
    tau_save_idx(a) = idx;
end

rows = {};

for a = 1:length(taus_to_be_saved)
    
    tau_val = taus_to_be_saved(a);
    k = tau_save_idx(a);
    
    for jj = 1:length(save_idx)
        
        j = save_idx(jj);
        
        rows(end+1, :) = { ...
            tau_val, ...
            char(x_var_names(j)), ...
            round(beta_bondell(j, k), 3), ...
            round(beta_unique(j, k), 3) ...
            }; %#ok<SAGROW>
    end
end

coef_table = cell2table(rows, ...
    'VariableNames', { ...
        'tau', ...
        'variable', ...
        'NCQR_slope', ...
        'UNIQUE_slope' ...
    });

writetable(coef_table, out_file);

fprintf('\nSaved coefficient table:\n%s\n', out_file);
disp(coef_table);