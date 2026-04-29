clear; clc;

%% =======================
%  USER OPTIONS
%% =======================
bootdir = 'Bootstrap for CI';
use_dataset = 2; % must be 1 / 2 / 3; 2 = main analysis data
Num_bootstrap = 500;
CI_level = 0.95;

% Quantile grid (must match bootstrap run)
tau_grid = 0.1:0.05:0.9;
tau_grid = tau_grid(:)';

alpha = 1 - CI_level;
q_lower = alpha/2;
q_upper = 1 - alpha/2;

%% =======================
%  READ ONE FILE TO GET DIMENSIONS
%% =======================
first_orig = fullfile(bootdir, ...
    sprintf('UNIQUE_boot_rep_%02d_set_%d_orig.csv', 1, use_dataset));

first_scaled = fullfile(bootdir, ...
    sprintf('UNIQUE_boot_rep_%02d_set_%d_scaled.csv', 1, use_dataset));

if ~isfile(first_orig)
    error('First original-scale bootstrap CSV not found: %s', first_orig);
end

if ~isfile(first_scaled)
    error('First scaled-space bootstrap CSV not found: %s', first_scaled);
end

coef_orig_1 = readmatrix(first_orig);
coef_scaled_1 = readmatrix(first_scaled);

[nrow_orig, ncol_orig] = size(coef_orig_1);
[nrow_scaled, ncol_scaled] = size(coef_scaled_1);

if ncol_orig ~= length(tau_grid)
    error('Original-scale CSV columns do not match length(tau_grid).');
end

if ncol_scaled ~= length(tau_grid)
    error('Scaled-space CSV columns do not match length(tau_grid).');
end

%% =======================
%  PREALLOCATE BOOT ARRAYS
%% =======================
boot_array_orig = NaN(nrow_orig, ncol_orig, Num_bootstrap);
boot_array_scaled = NaN(nrow_scaled, ncol_scaled, Num_bootstrap);

%% =======================
%  LOAD ALL BOOTSTRAP FILES
%% =======================
for rep = 1:Num_bootstrap

    infile_orig = fullfile(bootdir, ...
        sprintf('UNIQUE_boot_rep_%02d_set_%d_orig.csv', rep, use_dataset));

    infile_scaled = fullfile(bootdir, ...
        sprintf('UNIQUE_boot_rep_%02d_set_%d_scaled.csv', rep, use_dataset));

    if ~isfile(infile_orig)
        error('Missing original-scale bootstrap CSV: %s', infile_orig);
    end

    if ~isfile(infile_scaled)
        error('Missing scaled-space bootstrap CSV: %s', infile_scaled);
    end

    tmp_orig = readmatrix(infile_orig);
    tmp_scaled = readmatrix(infile_scaled);

    if ~isequal(size(tmp_orig), [nrow_orig, ncol_orig])
        error('Size mismatch in original-scale file: %s', infile_orig);
    end

    if ~isequal(size(tmp_scaled), [nrow_scaled, ncol_scaled])
        error('Size mismatch in scaled-space file: %s', infile_scaled);
    end

    boot_array_orig(:, :, rep) = tmp_orig;
    boot_array_scaled(:, :, rep) = tmp_scaled;
end

%% =======================
%  COMPUTE PERCENTILE CIs
%% =======================
ci_orig_lower = quantile(boot_array_orig, q_lower, 3);
ci_orig_upper = quantile(boot_array_orig, q_upper, 3);

ci_scaled_lower = quantile(boot_array_scaled, q_lower, 3);
ci_scaled_upper = quantile(boot_array_scaled, q_upper, 3);

%% =======================
%  SAVE CI CSV FILES
%% =======================
outfile_orig_lower = fullfile(bootdir, ...
    sprintf('UNIQUE_CI_lower_%0.2f_set_%d_orig.csv', CI_level, use_dataset));

outfile_orig_upper = fullfile(bootdir, ...
    sprintf('UNIQUE_CI_upper_%0.2f_set_%d_orig.csv', CI_level, use_dataset));

outfile_scaled_lower = fullfile(bootdir, ...
    sprintf('UNIQUE_CI_lower_%0.2f_set_%d_scaled.csv', CI_level, use_dataset));

outfile_scaled_upper = fullfile(bootdir, ...
    sprintf('UNIQUE_CI_upper_%0.2f_set_%d_scaled.csv', CI_level, use_dataset));

writematrix(ci_orig_lower, outfile_orig_lower);
writematrix(ci_orig_upper, outfile_orig_upper);
writematrix(ci_scaled_lower, outfile_scaled_lower);
writematrix(ci_scaled_upper, outfile_scaled_upper);


%% =======================
%  READ UNIQUE MATRIX-STYLE CSVs FROM 'Output'
%% =======================
outdir = 'Output';
sel_thr = 0.1;

file_scaled_est = fullfile(outdir, ...
    sprintf('UNIQUE_coeff_scaled_matrix_set_%d.csv', use_dataset));

file_orig_est = fullfile(outdir, ...
    sprintf('UNIQUE_coeff_original_matrix_set_%d.csv', use_dataset));

C_scaled = readcell(file_scaled_est);
C_orig   = readcell(file_orig_est);

col_names = string(C_scaled(1, 2:end));
row_names = string(C_scaled(2:end, 1));

coef_scaled_est_full = cell2mat(C_scaled(2:end, 2:end));
coef_orig_est_full   = cell2mat(C_orig(2:end, 2:end));

%% =======================
%  SPLIT INTERCEPT + VARIABLES
%% =======================
intercept_name = row_names(1);

var_names = row_names(2:end);

coef_scaled_est = coef_scaled_est_full(2:end, :);
coef_orig_est   = coef_orig_est_full(2:end, :);

ci_scaled_lower_vars = ci_scaled_lower(2:end, :);
ci_scaled_upper_vars = ci_scaled_upper(2:end, :);

ci_orig_lower_vars = ci_orig_lower(2:end, :);
ci_orig_upper_vars = ci_orig_upper(2:end, :);

%% =======================
%  VARIABLE SELECTION (SCALED)
%% =======================
mean_abs_scaled_est = mean(abs(coef_scaled_est), 2);
sel_idx = find(mean_abs_scaled_est > sel_thr);

%% =======================
%  BUILD OUTPUT (INTERCEPT + SELECTED)
%% =======================
nsel = length(sel_idx);
Ksel = size(coef_scaled_est_full, 2);

% +1 for header, +1 for intercept
nrows_total = 1 + 1 + nsel;

scaled_display = cell(nrows_total, Ksel + 1);
orig_display   = cell(nrows_total, Ksel + 1);

% Header row
scaled_display(1,1) = {'Variable'};
orig_display(1,1)   = {'Variable'};
scaled_display(1,2:end) = cellstr(col_names);
orig_display(1,2:end)   = cellstr(col_names);

%% -----------------------
%  INTERCEPT ROW (ALWAYS)
%% -----------------------
scaled_display{2,1} = char(intercept_name);
orig_display{2,1}   = char(intercept_name);

for k = 1:Ksel
    est_s = coef_scaled_est_full(1, k);
    lb_s  = ci_scaled_lower(1, k);
    ub_s  = ci_scaled_upper(1, k);

    est_o = coef_orig_est_full(1, k);
    lb_o  = ci_orig_lower(1, k);
    ub_o  = ci_orig_upper(1, k);

    scaled_display{2, k+1} = sprintf('%.3f (%.3f, %.3f)', est_s, lb_s, ub_s);
    orig_display{2, k+1}   = sprintf('%.4f (%.4f, %.4f)', est_o, lb_o, ub_o);
end

%% -----------------------
%  SELECTED VARIABLES
%% -----------------------
for ii = 1:nsel
    j = sel_idx(ii);
    row_out = ii + 2;   % after header + intercept

    scaled_display{row_out,1} = char(var_names(j));
    orig_display{row_out,1}   = char(var_names(j));

    for k = 1:Ksel
        est_s = coef_scaled_est(j, k);
        lb_s  = ci_scaled_lower_vars(j, k);
        ub_s  = ci_scaled_upper_vars(j, k);

        est_o = coef_orig_est(j, k);
        lb_o  = ci_orig_lower_vars(j, k);
        ub_o  = ci_orig_upper_vars(j, k);

        scaled_display{row_out, k+1} = sprintf('%.3f (%.3f, %.3f)', est_s, lb_s, ub_s);
        orig_display{row_out, k+1}   = sprintf('%.4f (%.4f, %.4f)', est_o, lb_o, ub_o);
    end
end

%% =======================
%  SAVE
%% =======================
outfile_scaled_summary = fullfile(outdir, ...
    sprintf('UNIQUE_selected_BOOT_CI_%d_summary_scaled_set_%d.csv',100*CI_level, use_dataset));

outfile_orig_summary = fullfile(outdir, ...
    sprintf('UNIQUE_selected_BOOT_CI_%d_summary_original_set_%d.csv',100*CI_level, use_dataset));

writecell(scaled_display, outfile_scaled_summary);
writecell(orig_display, outfile_orig_summary);

% 
%     %% =======================
%     %  SAVE TO 'Output'
%     %% =======================
%     outfile_scaled_summary = fullfile(outdir, ...
%         sprintf('UNIQUE_selected_BOOT_CI_%d_summary_scaled_set_%d.csv', 100*CI_level, use_dataset));
% 
%     outfile_orig_summary = fullfile(outdir, ...
%         sprintf('UNIQUE_selected_BOOT_CI_%d_summary_original_set_%d.csv', 100*CI_level, use_dataset));
% 
%     writecell(scaled_display, outfile_scaled_summary);
%     writecell(orig_display, outfile_orig_summary);
% end