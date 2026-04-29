function summary = summarize_post_BONDLL_metrics(results, opts)
% =========================================================================
% summarize_post_BONDLL_metrics
%
% Computes post-fit summary metrics for BONDELL only.
%
% Metrics returned:
%   - TPR
%   - FPR
%   - MCC
%   - FDR
%   - beta_RMSE
%   - Q_RMSE
%   - ratio_same
%   - ratio_opposite
%   - n_true
%   - comp_time
%
% REQUIRED IN opts FOR TRUTH-BASED METRICS:
%   opts.beta0_true_grid
%   opts.beta_true_grid
%   opts.true_support
%   opts.sel_thr
%
% INPUTS
%   results : output struct from BONDLL_NCQR(...)
%   opts    : options struct
%
% OUTPUT
%   summary : struct with field
%       .BONDELL
% =========================================================================

    % ---------------------------------------------------------
    % Check required truth inputs
    % ---------------------------------------------------------
    req_fields = {'beta0_true_grid', 'beta_true_grid', 'true_support', 'sel_thr'};
    for ii = 1:numel(req_fields)
        if ~isfield(opts, req_fields{ii})
            error('summarize_post_BONDLL_metrics: opts.%s is required.', req_fields{ii});
        end
    end

    % ---------------------------------------------------------
    % Extract basics
    % ---------------------------------------------------------
    if isfield(results, 'X_original')
        X = results.X_original;
    else
        X = results.X;
    end

    tau_grid = results.tau_grid;
    [n, p] = size(X);
    K = numel(tau_grid);

    sel_thr = opts.sel_thr;

    beta0_true_grid = opts.beta0_true_grid(:);   % K x 1
    beta_true_grid  = opts.beta_true_grid;       % p x K
    true_support    = logical(opts.true_support);

    beta0_hat_bondell = results.beta0_hat_bondell;   % K x 1
    beta_hat_bondell  = results.beta_hat_bondell;    % p x K

    % ---------------------------------------------------------
    % Build TRUE mask on (j,k) grid
    % ---------------------------------------------------------
    if ~isequal(size(true_support), [p, K])
        error('opts.true_support must be a p x K matrix.');
    end

    true_mask = logical(true_support);

    % ---------------------------------------------------------
    % Selection metrics
    % ---------------------------------------------------------
    sel_mask_bondell = abs(beta_hat_bondell) > sel_thr;

    [TPR_bondell, FPR_bondell, FDR_bondell, MCC_bondell] = ...
        compute_rates_coord(sel_mask_bondell, true_mask);

    % ---------------------------------------------------------
    % RMSE on coefficients
    % ---------------------------------------------------------
    beta_full_true    = [beta0_true_grid'; beta_true_grid];
    beta_full_bondell = [beta0_hat_bondell'; beta_hat_bondell];

    beta_RMSE_bondell = sqrt(mean((beta_full_bondell(:) - beta_full_true(:)).^2));

    % ---------------------------------------------------------
    % RMSE on fitted quantile surface evaluated at observed X
    % ---------------------------------------------------------
    Q_true_all    = ones(n,1) * beta0_true_grid'     + X * beta_true_grid;
    Q_bondell_all = ones(n,1) * beta0_hat_bondell'   + X * beta_hat_bondell;

    Q_RMSE_bondell = sqrt(mean((Q_bondell_all(:) - Q_true_all(:)).^2));

    % ---------------------------------------------------------
    % Sign agreement relative to truth, restricted to true active coords
    % ---------------------------------------------------------
    sign_stats_bondell = sign_agreement_coord_truth( ...
        beta_hat_bondell, beta_true_grid, true_mask, sel_thr);

    time_bondell = results.time_bondell;

    % ---------------------------------------------------------
    % Store summary
    % ---------------------------------------------------------
    summary = struct();

    summary.BONDELL = struct();
    summary.BONDELL.TPR            = TPR_bondell;
    summary.BONDELL.FPR            = FPR_bondell;
    summary.BONDELL.MCC            = MCC_bondell;
    summary.BONDELL.FDR            = FDR_bondell;
    summary.BONDELL.beta_RMSE      = beta_RMSE_bondell;
    summary.BONDELL.Q_RMSE         = Q_RMSE_bondell;
    summary.BONDELL.ratio_same     = sign_stats_bondell.ratio_same;
    summary.BONDELL.ratio_opposite = sign_stats_bondell.ratio_opposite;
    summary.BONDELL.n_true         = sign_stats_bondell.n_true;
    summary.BONDELL.comp_time      = time_bondell;

    % Optional extra counts
    summary.BONDELL.n_same         = sign_stats_bondell.n_same;
    summary.BONDELL.n_opposite     = sign_stats_bondell.n_opposite;
    summary.BONDELL.n_zero         = sign_stats_bondell.n_zero;
end