function opts = fill_default_opts(opts, p, K)

if ~isfield(opts, 'seed'), opts.seed = 1; end
if ~isfield(opts, 'verbose_crossfit'), opts.verbose_crossfit = 1; end
if ~isfield(opts, 'verbose_stage_2_tuning'), opts.verbose_stage_2_tuning = 1; end
if ~isfield(opts, 'verbose_stage_2_init'), opts.verbose_stage_2_init = 1; end
if ~isfield(opts, 'verbose_stage_3'), opts.verbose_stage_3 = 1; end


if ~isfield(opts, 'num_folds_for_crossfit'), opts.num_folds_for_crossfit = 10; end
if ~isfield(opts, 'num_folds_for_final_lambda'), opts.num_folds_for_final_lambda = 5; end
if opts.num_folds_for_final_lambda < 1 || floor(opts.num_folds_for_final_lambda) ~= opts.num_folds_for_final_lambda
    error('opts.num_folds_for_final_lambda must be a positive integer.');
end


if ~isfield(opts, 'eps_w'), opts.eps_w = 1e-6; end
if ~isfield(opts, 't_row'), opts.t_row = 0.05; end
if ~isfield(opts, 'gate_thr'), opts.gate_thr = 0.05; end

if ~isfield(opts, 'lambda_grid'), opts.lambda_grid = sort(logspace(-6, 2, 20)); end
if ~isfield(opts, 'lambda_grid_QRLASSO'), opts.lambda_grid_QRLASSO = sort(logspace(-5, 5, 20)); end

if ~isfield(opts, 'plot_cv_path'), opts.plot_cv_path = 1; end

if ~isfield(opts, 'run_qrlasso'), opts.run_qrlasso = 1; end
if ~isfield(opts, 'num_folds_qrlasso'), opts.num_folds_qrlasso = 5; end


if ~isfield(opts, 'use_fixed_lambda_opt'), opts.use_fixed_lambda_opt = 0; end

if ~isfield(opts, 'rand_feasible_initiation'), opts.rand_feasible_initiation = 0; end

% =========================================================
% Optional initialization stabilizers
% =========================================================
if ~isfield(opts, 'theta_cap'), opts.theta_cap = Inf; end
if ~isfield(opts, 'beta0_contrib_cap'), opts.beta0_contrib_cap = Inf; end

if ~isscalar(opts.theta_cap) || ~isnumeric(opts.theta_cap) || isnan(opts.theta_cap) || opts.theta_cap <= 0
    error('opts.theta_cap must be a positive scalar or Inf.');
end

if ~isscalar(opts.beta0_contrib_cap) || ~isnumeric(opts.beta0_contrib_cap) || isnan(opts.beta0_contrib_cap) || opts.beta0_contrib_cap <= 0
    error('opts.beta0_contrib_cap must be a positive scalar or Inf.');
end

% =========================================================
% Sign mask
% =========================================================
if ~isfield(opts, 'same_sign_mask') || isempty(opts.same_sign_mask)
    opts.same_sign_mask = true(p, K);
end

if ~isequal(size(opts.same_sign_mask), [p, K])
    error('opts.same_sign_mask must be p x K.');
end

opts.same_sign_mask = logical(opts.same_sign_mask);

% =========================================================
% Domain magnitude controls box constraints internally
% =========================================================
if ~isfield(opts, 'domain_magnitude') || isempty(opts.domain_magnitude)
    opts.domain_magnitude = 100;
end

if ~isscalar(opts.domain_magnitude) || ~isfinite(opts.domain_magnitude) || opts.domain_magnitude <= 0
    error('opts.domain_magnitude must be a positive finite scalar.');
end

M = opts.domain_magnitude;
npar = K + p*K;

lb_theta0 = -M * ones(K,1);
ub_theta0 =  M * ones(K,1);

lb_theta_mat = -M * ones(p, K);
ub_theta_mat =  M * ones(p, K);

lb_theta_mat(opts.same_sign_mask) = 0;

opts.lb = [lb_theta0; lb_theta_mat(:)];
opts.ub = [ub_theta0; ub_theta_mat(:)];

if length(opts.lb) ~= npar || length(opts.ub) ~= npar
    error('Internal error: opts.lb and opts.ub must both have length K + p*K.');
end

if any(opts.lb > opts.ub)
    error('Internal error: some lower bounds exceed upper bounds.');
end



if ~isfield(opts, 'params')
    params = struct();
    params.s_init        = 0.02;
    params.s_inc         = 2;
    params.s_dec         = 2;
    params.p_inc         = 2;
    params.p_dec         = 2;
    params.m             = 500;
    params.n_hit_and_run = 5;
    params.T             = round(3000 * log(K + p*K));
    params.M             = 10 * (K + p*K);
    params.epsilon       = 1e-12;
    params.feas_tol      = 1e-8;  % keep less than 10^(-10)
    params.c_log         = 0.001 * log(K + p*K);
    opts.params = params;
end

if ~isfield(opts, 'outdir'), opts.outdir = 'Output'; end
if ~isfield(opts, 'sel_thr'), opts.sel_thr = 1e-1; end
end