clear; clc;
addpath('./BONDELL source funs/');
addpath('./Data/');

outdir = 'Output';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

datadir = 'Data';

%% =======================
%  DATA LOADING
%% =======================
Num_exps = 20;
n = 500;
p = 100;
p_true = 4;
design_type = 'corr'; % 'corr'/ 'indep'

% Load tau grid
fname_tau = fullfile(datadir, ...
    sprintf('Data_tau_grid_p_%d_n_%d_design_%s.csv', p, n, design_type));
tau_grid = readmatrix(fname_tau);
tau_grid = tau_grid(:)';   % ensure row vector
K = length(tau_grid);

do_post_computation_plot = 0; %#ok<NASGU>
plot_upto_p = min(p,7); %#ok<NASGU>   % kept for stylistic similarity; not used

%% =======================
%  BONDELL options
%% =======================

opts = struct();

% ---------------------------------------------------------
% Load true parameters
% ---------------------------------------------------------
fname_support = fullfile(datadir, ...
    sprintf('Data_TRUE_support_p_%d_n_%d_design_%s.csv', p, n, design_type));

fname_beta = fullfile(datadir, ...
    sprintf('Data_TRUE_beta_p_%d_n_%d_design_%s.csv', p, n, design_type));

fname_beta0 = fullfile(datadir, ...
    sprintf('Data_TRUE_beta0_p_%d_n_%d_design_%s.csv', p, n, design_type));

true_support    = logical(readmatrix(fname_support));  % p x K
beta_true_grid  = readmatrix(fname_beta);              % p x K
beta0_true_grid = readmatrix(fname_beta0);             % K x 1
beta0_true_grid = beta0_true_grid(:);                  % ensure column

% ---------------------------------------------------------
% Attach to opts
% ---------------------------------------------------------
opts.beta0_true_grid = beta0_true_grid;
opts.beta_true_grid  = beta_true_grid;
opts.true_support    = true_support;

%%% Threshold for TPR/FPR etc metric calculations
opts.sel_thr         = 1e-1;

%%% BONDELL fitting options
opts.verbose         = 1;
opts.scale_X_to_unit = 1;
opts.linprog_display = 'none';

% Optional: uncomment if you want custom linprog options
% opts.linprog_options = optimoptions('linprog', ...
%     'Display', 'none', ...
%     'Algorithm', 'dual-simplex');

parfor rep_num = 1:Num_exps
    rng(rep_num);

    fprintf('--------------------------------------------\n');
    fprintf('BONDELL Simulation\n');
    fprintf('Performing rep = %d, n = %d, p = %d, design = %s \n', ...
        rep_num, n, p, design_type);
    fprintf('--------------------------------------------\n');

    % =====================================================
    % Load data matrix [Y, X]
    % =====================================================
    fname_data = fullfile(datadir, ...
        sprintf('Data_p_%d_n_%d_design_%s_rep_%d.csv', p, n, design_type, rep_num));

    YX = readmatrix(fname_data);

    Y = YX(:,1);
    X = YX(:,2:end);

    % =====================================================
    % Load true support / beta / beta0
    % =====================================================
    fname_support = fullfile(datadir, ...
        sprintf('Data_TRUE_support_p_%d_n_%d_design_%s.csv', p, n, design_type));

    fname_beta = fullfile(datadir, ...
        sprintf('Data_TRUE_beta_p_%d_n_%d_design_%s.csv', p, n, design_type));

    fname_beta0 = fullfile(datadir, ...
        sprintf('Data_TRUE_beta0_p_%d_n_%d_design_%s.csv', p, n, design_type));

    true_support    = readmatrix(fname_support);
    beta_true_grid  = readmatrix(fname_beta);
    beta0_true_grid = readmatrix(fname_beta0);

    % Ensure expected shapes
    beta0_true_grid = beta0_true_grid(:);   % K x 1
    true_support    = logical(true_support); % p x K

    % Optional safety checks
    if size(X,1) ~= n || size(X,2) ~= p
        error('Loaded X has incompatible dimensions.');
    end

    if length(Y) ~= n
        error('Loaded Y has incompatible length.');
    end

    if ~isequal(size(true_support), [p, K])
        error('Loaded true_support must be p x K.');
    end

    if ~isequal(size(beta_true_grid), [p, K])
        error('Loaded beta_true_grid must be p x K.');
    end

    if length(beta0_true_grid) ~= K
        error('Loaded beta0_true_grid must have length K.');
    end

    %% =======================
    %  Fit BONDELL
    %% =======================
    results = BONDLL_NCQR(X, Y, tau_grid, opts);

    do_truth_summary = isfield(opts, 'beta0_true_grid') && ...
        isfield(opts, 'beta_true_grid')  && ...
        isfield(opts, 'true_support')    && ...
        isfield(opts, 'sel_thr');

    if do_truth_summary
        summary = summarize_post_BONDLL_metrics(results, opts);
        print_post_BONDLL_metrics(summary);

        % If save_summary_to_csv is generic and takes a method label:
        save_summary_to_csv_BONDLL(summary, p, n, design_type, rep_num, outdir);

        % Otherwise replace the above line by a Bondell-specific saver, e.g.
        % save_summary_to_csv_BONDLL(summary, p, n, design_type, rep_num, outdir);
    end

end