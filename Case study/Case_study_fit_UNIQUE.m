clear; clc;
addpath('./UNIQUE source funs/');
addpath('./SAPS/');
rng(1)

outdir = 'Output';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

datadir_real = 'PPMI data';

%% =======================
%  USER OPTION
%% =======================
use_dataset = 2;   % must be 1 / 2 / 3; 2 = main analysis data, 3 = main analysis vars minus cognitive marker vars
eps_plot = 5e-1;
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

% If first column is just an index column created during CSV export, remove it
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

% If first column is just an index column created during CSV export, remove it
if width(TY) >= 2
    first_col_y = TY{:,1};
    if isnumeric(first_col_y) && isequal(first_col_y(:), (1:height(TY))')
        TY(:,1) = [];
    end
end

Y = table2array(TY);

% Force Y to be n x 1
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
        warning('X appears transposed relative to Y. Transposing X.');
        X = X';
    else
        error('X and Y have incompatible dimensions.');
    end
end

% Define n and p only after reading real data
[n, p] = size(X);

if length(x_var_names) ~= p
    error('Number of X variable names does not match number of X columns.');
end

fprintf('--------------------------------------------\n');
fprintf('UNIQUE + QRLASSO Real Data Run\n');
fprintf('Using dataset option = %d\n', use_dataset);
fprintf('n = %d, p = %d, K = %d quantiles\n', n, p, K);
fprintf('--------------------------------------------\n');

%% =======================
%  UNIQUE + SAPS parameters
%% =======================

%% =======================
%  Selective sign mask based on prior scientific knowledge
%% =======================

% Start with all FALSE: unconstrained by default
same_sign_mask = false(p, K);

% Conservative constrained-variable list by dataset
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
            "updrs2_score","updrs3_score",
            ];

    otherwise
        error('Unsupported use_dataset.');
end

% Turn on mask for available variables only
for j = 1:p
    if any(strcmp(x_var_names(j), constrained_vars))
        same_sign_mask(j, :) = true;
    end
end

fprintf('\nSelective same_sign_mask activated for %d of %d predictors.\n', ...
    sum(any(same_sign_mask, 2)), p);

disp('Variables with sign constraints:');
disp(x_var_names(any(same_sign_mask, 2)));

%%%%

%%% Tuning UNIQUE
t_row = 0.05;                   % same as default, harmonic mean of row_importance cut-off
gate_thr = 0.05;                % same as default, active row cutoff
num_folds_for_crossfit = 10;    % same as default
num_folds_for_final_lambda = 5; % same as default
lambda_grid_UNIQUE = sort(logspace(-5, 5, 20)); % same as default

%%% Tuning UNIQUE warmstart
rand_feasible_initiation = 0;   % same as default
theta_cap                = 20;  % default = Inf
beta0_contrib_cap        = 20;  % default = Inf

%%% Tuning QR-LASSO
num_folds_qrlasso    = 5;
lambda_grid_QRLASSO  = sort(logspace(-5, 5, 20));

%% =======================
%  UNIQUE and SAPS options
%% =======================

opts = struct();

%%% IMPORTANT: run both UNIQUE and QR-LASSO
opts.run_qrlasso = 1;
opts.num_folds_qrlasso   = num_folds_qrlasso;
opts.lambda_grid_QRLASSO = lambda_grid_QRLASSO;

%%% Print results
opts.verbose_crossfit               = 1;
opts.verbose_stage_2_init           = 1;
opts.verbose_stage_2_tuning         = 1;
opts.verbose_stage_3                = 1;

%%% UNIQUE parameters
opts.same_sign_mask                 = same_sign_mask;
opts.num_folds_for_crossfit         = num_folds_for_crossfit;
opts.num_folds_for_final_lambda     = num_folds_for_final_lambda;
opts.lambda_grid                    = lambda_grid_UNIQUE;

%%% warm starting point tuning
opts.rand_feasible_initiation       = rand_feasible_initiation;
opts.t_row                          = t_row;
opts.gate_thr                       = gate_thr;
opts.theta_cap                      = theta_cap;
opts.beta0_contrib_cap              = beta0_contrib_cap;

%%% SAPS tuning
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
%  Fit UNIQUE once
%% =======================
% Since opts.run_qrlasso = 1, this also runs QR-LASSO internally
results = UNIQUE(X, Y, tau_grid, opts);

%% =======================
%  Shared naming objects
%% =======================
row_names = ["intercept"; x_var_names(:)];
col_names = strcat("tau_", string(tau_grid));

%% =========================================================
%  FINAL UNIQUE COEFFICIENTS
%% =========================================================
% The FINAL UNIQUE fitted coefficients are:
%   results.beta0_hat_unilasso   : K x 1
%   results.beta_hat_unilasso    : p x K
%
% These are in ORIGINAL X scale after back-transformation, if scaling
% to [0,1] was used inside UNIQUE.

if ~isfield(results, 'beta0_hat_unilasso')
    error('results.beta0_hat_unilasso not found.');
end

if ~isfield(results, 'beta_hat_unilasso')
    error('results.beta_hat_unilasso not found.');
end

beta0_orig_unique = results.beta0_hat_unilasso;
beta_orig_unique  = results.beta_hat_unilasso;

beta0_orig_unique = beta0_orig_unique(:);   % K x 1

if ~isequal(size(beta_orig_unique), [p, K])
    error('results.beta_hat_unilasso must be of size p x K.');
end

if length(beta0_orig_unique) ~= K
    error('results.beta0_hat_unilasso must have length K.');
end

%% =========================================================
%  RECONSTRUCT UNIQUE INTERNAL SCALED-[0,1] COEFFICIENTS
%% =========================================================

if ~isfield(results, 'X_min') || ~isfield(results, 'X_range')
    error('results.X_min / results.X_range not found.');
end

X_min   = results.X_min;
X_range = results.X_range;

beta_scaled_unique = results.beta_hat_unilasso_raw;
beta0_scaled_unique = results.beta0_hat_unilasso_raw;

%% =========================================================
%  BUILD UNIQUE COEFFICIENT SUMMARY TABLES
%% =========================================================

coef_summary_orig_unique = [beta0_orig_unique'; beta_orig_unique];
coef_summary_scaled_unique = [beta0_scaled_unique'; beta_scaled_unique];

coef_summary_orig_table_unique = array2table( ...
    coef_summary_orig_unique, ...
    'VariableNames', matlab.lang.makeValidName(cellstr(col_names)), ...
    'RowNames', cellstr(row_names) ...
);

coef_summary_scaled_table_unique = array2table( ...
    coef_summary_scaled_unique, ...
    'VariableNames', matlab.lang.makeValidName(cellstr(col_names)), ...
    'RowNames', cellstr(row_names) ...
);

disp('==========================================================');
disp('FINAL UNIQUE Coefficients (ORIGINAL X scale)');
disp('==========================================================');
disp(coef_summary_orig_table_unique);

disp('==========================================================');
disp('FINAL UNIQUE Coefficients (INTERNAL SCALED [0,1] X space)');
disp('==========================================================');
disp(coef_summary_scaled_table_unique);

%% =========================================================
%  SAVE UNIQUE MATRIX-STYLE CSVs
%% =========================================================

coef_matrix_outfile_orig_unique = fullfile(outdir, ...
    sprintf('UNIQUE_coeff_original_matrix_set_%d.csv', use_dataset));

writecell( ...
    [ ...
      [{'Variable'}, cellstr(strrep(col_names, '_', '='))]; ...
      [cellstr(row_names), num2cell(coef_summary_orig_unique)] ...
    ], ...
    coef_matrix_outfile_orig_unique ...
);

coef_matrix_outfile_scaled_unique = fullfile(outdir, ...
    sprintf('UNIQUE_coeff_scaled_matrix_set_%d.csv', use_dataset));

writecell( ...
    [ ...
      [{'Variable'}, cellstr(strrep(col_names, '_', '='))]; ...
      [cellstr(row_names), num2cell(coef_summary_scaled_unique)] ...
    ], ...
    coef_matrix_outfile_scaled_unique ...
);

fprintf('Original-scale UNIQUE matrix CSV saved to:\n%s\n', coef_matrix_outfile_orig_unique);
fprintf('Scaled-space UNIQUE matrix CSV saved to:\n%s\n', coef_matrix_outfile_scaled_unique);

%% =========================================================
%  PLOT UNIQUE SELECTED COEFFICIENT CURVES IN SCALED SPACE
%% =========================================================
% Selection rule:
% keep variables j such that sum_k |beta_j(tau_k)| > eps_plot
%
% IMPORTANT:
% These are the INTERNAL scaled-[0,1]-space coefficients, not the
% back-transformed original-scale coefficients.
%
% Intercept is plotted separately with its own Y-axis scale.
% All selected beta_j(tau) panels share the SAME Y-axis range so that
% magnitudes are visually comparable across predictors.



sel_plot_unique = sum(abs(beta_scaled_unique), 2) > eps_plot;
sel_idx_unique  = find(sel_plot_unique);
num_sel_unique  = length(sel_idx_unique);

fprintf('\nNumber of selected variables for UNIQUE scaled-space beta-curve plotting: %d\n', num_sel_unique);

if num_sel_unique == 0
    
    fprintf('No UNIQUE variables satisfied sum(abs(beta_j(.))) > %.2e\n', eps_plot);
    
else
    
    % ---------------------------------------------------------
    % Compute common Y-limits across all selected beta_j curves
    % ---------------------------------------------------------
    beta_sel_unique = beta_scaled_unique(sel_idx_unique, :);
    
    ymin_unique = min(beta_sel_unique(:));
    ymax_unique = max(beta_sel_unique(:));
    
    yrange_unique = ymax_unique - ymin_unique;
    if yrange_unique < 1e-12
        yrange_unique = 1;
    end
    ymin_unique = ymin_unique - 0.05 * yrange_unique;
    ymax_unique = ymax_unique + 0.05 * yrange_unique;
    
    % ---------------------------------------------------------
    % Number of panels:
    % 1 for intercept + one for each selected variable
    % ---------------------------------------------------------
    num_panels_unique = num_sel_unique + 1;
    
    nrow_plot_unique = 2;
    ncol_plot_unique = ceil(num_panels_unique / nrow_plot_unique);
    
    figure('Color', 'w', 'Position', [100, 100, 320*ncol_plot_unique, 700]);
    
    % ---------------------------------------------------------
    % Panel 1: intercept curve
    % ---------------------------------------------------------
    subplot(nrow_plot_unique, ncol_plot_unique, 1);
    plot(tau_grid, beta0_scaled_unique, '-o', 'LineWidth', 2, 'MarkerSize', 5);
    xlim([min(tau_grid), max(tau_grid)]);
    xlabel('\tau', 'FontSize', 11);
    ylabel('\beta_0(\tau)', 'FontSize', 11);
    title('Intercept (scaled X-space)', 'FontSize', 12, 'Interpreter', 'none');
    grid on;
    box on;
    set(gca, 'FontSize', 10);
    
    % ---------------------------------------------------------
    % Remaining panels: selected beta_j(tau) curves
    % All use the SAME Y-axis range
    % ---------------------------------------------------------
    for ii = 1:num_sel_unique
        j = sel_idx_unique(ii);
        
        subplot(nrow_plot_unique, ncol_plot_unique, ii + 1);
        plot(tau_grid, beta_scaled_unique(j, :), '-o', 'LineWidth', 2, 'MarkerSize', 5);
        
        ylim([ymin_unique, ymax_unique]);
        xlim([min(tau_grid), max(tau_grid)]);
        yline(0, '--k', 'LineWidth', 1);
        
        xlabel('\tau', 'FontSize', 11);
        ylabel(sprintf('\\beta_{%d}(\\tau)', j), 'FontSize', 11);
        title(char(x_var_names(j)), 'FontSize', 11, 'Interpreter', 'none');
        grid on;
        box on;
        set(gca, 'FontSize', 10);
    end
    
    sgtitle(sprintf(['UNIQUE coefficient curves in internal scaled [0,1] X-space ', ...
                     '(dataset %d, eps\\_plot = %.1e)'], ...
                     use_dataset, eps_plot), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    plot_outfile_unique = fullfile(outdir, ...
        sprintf('UNIQUE_scaled_beta_curves_set_%d.png', use_dataset));
    
    saveas(gcf, plot_outfile_unique);
    
    fprintf('UNIQUE scaled-space beta-curve plot saved to:\n%s\n', plot_outfile_unique);
    
end

%% =========================================================
%  FINAL QR-LASSO COEFFICIENTS
%% =========================================================
% Since opts.run_qrlasso = 1, UNIQUE.m stores:
%   results.beta0_qrlasso : K x 1
%   results.beta_qrlasso  : p x K
%
% These are already back-transformed to ORIGINAL X scale inside UNIQUE.m
% if internal scaling was used.

if ~isfield(results, 'beta0_qrlasso')
    error('results.beta0_qrlasso not found. Make sure opts.run_qrlasso = 1.');
end

if ~isfield(results, 'beta_qrlasso')
    error('results.beta_qrlasso not found. Make sure opts.run_qrlasso = 1.');
end

beta0_orig_qrlasso = results.beta0_qrlasso;
beta_orig_qrlasso  = results.beta_qrlasso;

beta0_orig_qrlasso = beta0_orig_qrlasso(:);   % K x 1

if ~isequal(size(beta_orig_qrlasso), [p, K])
    error('results.beta_qrlasso must be of size p x K.');
end

if length(beta0_orig_qrlasso) ~= K
    error('results.beta0_qrlasso must have length K.');
end

%% =========================================================
%  RECONSTRUCT QR-LASSO INTERNAL SCALED-[0,1] COEFFICIENTS
%% =========================================================

beta_scaled_qrlasso = zeros(p, K);
beta0_scaled_qrlasso = zeros(K, 1);

for k = 1:K
    beta_scaled_qrlasso(:, k) = beta_orig_qrlasso(:, k) .* X_range(:);
    beta0_scaled_qrlasso(k)   = beta0_orig_qrlasso(k) + sum(beta_orig_qrlasso(:, k)' .* X_min);
end

%% =========================================================
%  BUILD QR-LASSO COEFFICIENT SUMMARY TABLES
%% =========================================================

coef_summary_orig_qrlasso = [beta0_orig_qrlasso'; beta_orig_qrlasso];
coef_summary_scaled_qrlasso = [beta0_scaled_qrlasso'; beta_scaled_qrlasso];

coef_summary_orig_table_qrlasso = array2table( ...
    coef_summary_orig_qrlasso, ...
    'VariableNames', matlab.lang.makeValidName(cellstr(col_names)), ...
    'RowNames', cellstr(row_names) ...
);

coef_summary_scaled_table_qrlasso = array2table( ...
    coef_summary_scaled_qrlasso, ...
    'VariableNames', matlab.lang.makeValidName(cellstr(col_names)), ...
    'RowNames', cellstr(row_names) ...
);

disp('==========================================================');
disp('FINAL QR-LASSO Coefficients (ORIGINAL X scale)');
disp('==========================================================');
disp(coef_summary_orig_table_qrlasso);

disp('==========================================================');
disp('FINAL QR-LASSO Coefficients (INTERNAL SCALED [0,1] X space)');
disp('==========================================================');
disp(coef_summary_scaled_table_qrlasso);

%% =========================================================
%  SAVE QR-LASSO MATRIX-STYLE CSVs
%% =========================================================

coef_matrix_outfile_orig_qrlasso = fullfile(outdir, ...
    sprintf('QRLASSO_coeff_original_matrix_set_%d.csv', use_dataset));

writecell( ...
    [ ...
      [{'Variable'}, cellstr(strrep(col_names, '_', '='))]; ...
      [cellstr(row_names), num2cell(coef_summary_orig_qrlasso)] ...
    ], ...
    coef_matrix_outfile_orig_qrlasso ...
);

coef_matrix_outfile_scaled_qrlasso = fullfile(outdir, ...
    sprintf('QRLASSO_coeff_scaled_matrix_set_%d.csv', use_dataset));

writecell( ...
    [ ...
      [{'Variable'}, cellstr(strrep(col_names, '_', '='))]; ...
      [cellstr(row_names), num2cell(coef_summary_scaled_qrlasso)] ...
    ], ...
    coef_matrix_outfile_scaled_qrlasso ...
);

fprintf('Original-scale QRLASSO matrix CSV saved to:\n%s\n', coef_matrix_outfile_orig_qrlasso);
fprintf('Scaled-space QRLASSO matrix CSV saved to:\n%s\n', coef_matrix_outfile_scaled_qrlasso);

%% =========================================================
%  PLOT QR-LASSO SELECTED COEFFICIENT CURVES IN SCALED SPACE
%% =========================================================
% Apply the exact same plotting logic used for UNIQUE, but save under
% QRLASSO-specific file names.

sel_plot_qrlasso = sum(abs(beta_scaled_qrlasso), 2) > eps_plot;
sel_idx_qrlasso  = find(sel_plot_qrlasso);
num_sel_qrlasso  = length(sel_idx_qrlasso);

fprintf('\nNumber of selected variables for QRLASSO scaled-space beta-curve plotting: %d\n', num_sel_qrlasso);

if num_sel_qrlasso == 0
    
    fprintf('No QRLASSO variables satisfied sum(abs(beta_j(.))) > %.2e\n', eps_plot);
    
else
    
    % ---------------------------------------------------------
    % Compute common Y-limits across all selected beta_j curves
    % ---------------------------------------------------------
    beta_sel_qrlasso = beta_scaled_qrlasso(sel_idx_qrlasso, :);
    
    ymin_qrlasso = min(beta_sel_qrlasso(:));
    ymax_qrlasso = max(beta_sel_qrlasso(:));
    
    yrange_qrlasso = ymax_qrlasso - ymin_qrlasso;
    if yrange_qrlasso < 1e-12
        yrange_qrlasso = 1;
    end
    ymin_qrlasso = ymin_qrlasso - 0.05 * yrange_qrlasso;
    ymax_qrlasso = ymax_qrlasso + 0.05 * yrange_qrlasso;
    
    % ---------------------------------------------------------
    % Number of panels:
    % 1 for intercept + one for each selected variable
    % ---------------------------------------------------------
    num_panels_qrlasso = num_sel_qrlasso + 1;
    
    nrow_plot_qrlasso = 2;
    ncol_plot_qrlasso = ceil(num_panels_qrlasso / nrow_plot_qrlasso);
    
    figure('Color', 'w', 'Position', [100, 100, 320*ncol_plot_qrlasso, 700]);
    
    % ---------------------------------------------------------
    % Panel 1: intercept curve
    % ---------------------------------------------------------
    subplot(nrow_plot_qrlasso, ncol_plot_qrlasso, 1);
    plot(tau_grid, beta0_scaled_qrlasso, '-o', 'LineWidth', 2, 'MarkerSize', 5);
    xlim([min(tau_grid), max(tau_grid)]);
    xlabel('\tau', 'FontSize', 11);
    ylabel('\beta_0(\tau)', 'FontSize', 11);
    title('Intercept (scaled X-space)', 'FontSize', 12, 'Interpreter', 'none');
    grid on;
    box on;
    set(gca, 'FontSize', 10);
    
    % ---------------------------------------------------------
    % Remaining panels: selected beta_j(tau) curves
    % All use the SAME Y-axis range
    % ---------------------------------------------------------
    for ii = 1:num_sel_qrlasso
        j = sel_idx_qrlasso(ii);
        
        subplot(nrow_plot_qrlasso, ncol_plot_qrlasso, ii + 1);
        plot(tau_grid, beta_scaled_qrlasso(j, :), '-o', 'LineWidth', 2, 'MarkerSize', 5);
        
        ylim([ymin_qrlasso, ymax_qrlasso]);
        xlim([min(tau_grid), max(tau_grid)]);
        yline(0, '--k', 'LineWidth', 1);
        
        xlabel('\tau', 'FontSize', 11);
        ylabel(sprintf('\\beta_{%d}(\\tau)', j), 'FontSize', 11);
        title(char(x_var_names(j)), 'FontSize', 11, 'Interpreter', 'none');
        grid on;
        box on;
        set(gca, 'FontSize', 10);
    end
    
    sgtitle(sprintf(['QRLASSO coefficient curves in internal scaled [0,1] X-space ', ...
                     '(dataset %d, eps\\_plot = %.1e)'], ...
                     use_dataset, eps_plot), ...
            'FontSize', 14, 'FontWeight', 'bold');
    
    plot_outfile_qrlasso = fullfile(outdir, ...
        sprintf('QRLASSO_scaled_beta_curves_set_%d.png', use_dataset));
    
    saveas(gcf, plot_outfile_qrlasso);
    
    fprintf('QRLASSO scaled-space beta-curve plot saved to:\n%s\n', plot_outfile_qrlasso);
    
end