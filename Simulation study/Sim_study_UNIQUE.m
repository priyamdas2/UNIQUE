clear; clc;
addpath('./UNIQUE source funs/');
addpath('./SAPS/');
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
design_type = 'indep'; % 'corr'/ 'indep'
% Load tau grid
fname_tau = fullfile(datadir, ...
    sprintf('Data_tau_grid_p_%d_n_%d_design_%s.csv', p, n, design_type));
tau_grid = readmatrix(fname_tau);
tau_grid = tau_grid(:)';   % ensure row vector
K = length(tau_grid);


do_post_computation_plot = 0;
plot_upto_p = min(p,7);   % can be 1,2,...,p

%% UNIQUE + SAPS parameters

%%% UNIQUE mask
same_sign_mask  = true(p, K);   % MUST KEEP default = true(p, K); to apply custom mask, use the UNIQUE version in the case study

%%% Tuning UNIQUE
t_row = 0.05;                   % same as default, harmonic mean of row_importance cut-off, tune in (0.02,0.10)
gate_thr = 0.05;                % same as default, active row cutoff
num_folds_for_crossfit = 10;    % same as default
num_folds_for_final_lambda = 5; % same as default
lambda_grid_UNIQUE = sort(logspace(-5, 5, 20)); % same as default

%%% Tuning UNIQUE warmstart
rand_feasible_initiation = 0;   % same as default
theta_cap                = 20;  % default = Inf
beta0_contrib_cap        = 20;  % default = Inf

%%% Tuning QRLASSO
run_qrlasso          = 1;                           % same as default
num_folds_qrlasso    = 5;                           % same as default
lambda_grid_QRLASSO  = sort(logspace(-5, 5, 20));   % same as default

%%% Threshold for TPR/FPR etc metric calculations
sel_thr              = 1e-1;  % same as default, 

%% =======================
%  UNIQUE and SAPS options
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
opts.sel_thr         = sel_thr;

%%% QR LASSO parameter
opts.run_qrlasso                    = run_qrlasso;
opts.num_folds_qrlasso              = num_folds_qrlasso;
opts.lambda_grid_QRLASSO            = lambda_grid_QRLASSO;

%%% Print results
opts.verbose_crossfit               = 1; % same as default
opts.verbose_stage_2_init           = 1; % same as default
opts.verbose_stage_2_tuning         = 1; % same as default
opts.verbose_stage_3                = 1; % same as default


%%% UNIQUE parameters
opts.same_sign_mask                 = same_sign_mask; % % same as default, masking; true means all signs same as univariate sign
opts.num_folds_for_crossfit         = num_folds_for_crossfit;   % Number of folds for crossfitting in Stage 1
opts.num_folds_for_final_lambda     = num_folds_for_final_lambda;    % Number of folds for finding optimal lambda before final fit; 1 implies one 80/20 fold cv
opts.lambda_grid                    = lambda_grid_UNIQUE;


%%% warm starting point tuning
opts.rand_feasible_initiation       = rand_feasible_initiation; % same as default, keep it 0 for warm start
opts.t_row                          = t_row;  % harmonic mean of row_importance cut-off
opts.gate_thr                       = gate_thr; % active row cutoff
opts.theta_cap                      = theta_cap;    % Warm start cap on theta_cap
opts.beta0_contrib_cap              = beta0_contrib_cap;    % Warm start cap

%%% SAPS tuning

opts.domain_magnitude = 100;

params               = struct();  % ALL same as default
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


parfor rep_num = 1:Num_exps
    rng(rep_num);
    
    fprintf('--------------------------------------------\n');
    fprintf('UNIQUE Simulation\n');
    fprintf('Performing rep = %d, n = %d, p = %d, design = %s \n',rep_num, n, p, design_type);
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
    
    true_support   = readmatrix(fname_support);
    beta_true_grid = readmatrix(fname_beta);
    beta0_true_grid = readmatrix(fname_beta0);
    
    % Ensure expected shapes
    beta0_true_grid = beta0_true_grid(:);   % K x 1
    true_support = logical(true_support);   % p x K
    
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
    %  Fit UNIQUE
    %% =======================
    
    results = UNIQUE(X, Y, tau_grid, opts);
    
    
    do_truth_summary = isfield(opts, 'beta0_true_grid') && ...
        isfield(opts, 'beta_true_grid')  && ...
        isfield(opts, 'true_support')    && ...
        isfield(opts, 'sel_thr');
    
    if do_truth_summary
        summary = summarize_post_UNIQUE_metrics(results, opts);
        print_post_UNIQUE_metrics(summary);
        save_summary_to_csv(summary, p, n, design_type, rep_num, outdir);
    end
    
    
end
if do_post_computation_plot == 1
    prefix = sprintf('Output_plot_p_%d_n_%d_design_%s', p, n, design_type);
    plot_save_post_UNIQUE_outputs(results, opts, outdir, prefix, design_type, plot_upto_p);
end