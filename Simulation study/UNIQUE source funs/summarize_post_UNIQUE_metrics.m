function summary = summarize_post_UNIQUE_metrics(results, opts)
% =========================================================================
% summarize_post_UNIQUE_metrics
%
% Computes post-fit summary metrics for UNIQUE and, optionally, QR-LASSO.
%
% Metrics returned for each method:
%   - TPR
%   - FPR
%   - MCC
%   - FDR
%   - beta_RMSE
%   - Q_RMSE
%   - ratio_same
%   - ratio_opposite
%   - n_true
%   - comp. time
%
% QR-LASSO metrics are computed only if:
%   opts.run_qrlasso == 1
%
% REQUIRED IN opts FOR TRUTH-BASED METRICS:
%   opts.beta0_true_grid
%   opts.beta_true_grid
%   opts.true_support
%   opts.sel_thr
%
% INPUTS
%   results : output struct from UNIQUE(...)
%   opts    : options struct
%
% OUTPUT
%   summary : struct with fields
%       .UNIQUE
%       .QRLASSO   (only if available)
%
% =========================================================================

    % ---------------------------------------------------------
    % Check required truth inputs
    % ---------------------------------------------------------
    req_fields = {'beta0_true_grid', 'beta_true_grid', 'true_support', 'sel_thr'};
    for ii = 1:numel(req_fields)
        if ~isfield(opts, req_fields{ii})
            error('summarize_post_UNIQUE_metrics: opts.%s is required.', req_fields{ii});
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
    true_support    = opts.true_support;

    beta0_hat_unilasso = results.beta0_hat_unilasso;   % K x 1
    beta_hat_unilasso  = results.beta_hat_unilasso;    % p x K

    % ---------------------------------------------------------
    % Build TRUE mask on (j,k) grid
    % ---------------------------------------------------------
    if ~isequal(size(true_support), [p, K])
        error('opts.true_support must be a p x K matrix.');
    end
    
    true_mask = logical(true_support);

    % ---------------------------------------------------------
    % UNIQUE metrics
    % ---------------------------------------------------------
    sel_mask_unilasso = abs(beta_hat_unilasso) > sel_thr;

    [TPR_unilasso, FPR_unilasso, FDR_unilasso, MCC_unilasso] = ...
        compute_rates_coord(sel_mask_unilasso, true_mask);

    beta_full_true     = [beta0_true_grid'; beta_true_grid];
    beta_full_unilasso = [beta0_hat_unilasso'; beta_hat_unilasso];

    beta_RMSE_unilasso = sqrt(mean((beta_full_unilasso(:) - beta_full_true(:)).^2));

    Q_true_all     = ones(n,1) * beta0_true_grid'    + X * beta_true_grid;
    Q_unilasso_all = ones(n,1) * beta0_hat_unilasso' + X * beta_hat_unilasso;

    Q_RMSE_unilasso = sqrt(mean((Q_unilasso_all(:) - Q_true_all(:)).^2));
    time_unilasso   = results.time_unilasso;

    sign_stats_unilasso = sign_agreement_coord( ...
        beta_hat_unilasso, results.beta_univ_hat, true_mask, sel_thr);

    summary = struct();

    summary.UNIQUE = struct();
    summary.UNIQUE.TPR            = TPR_unilasso;
    summary.UNIQUE.FPR            = FPR_unilasso;
    summary.UNIQUE.MCC            = MCC_unilasso;
    summary.UNIQUE.FDR            = FDR_unilasso;
    summary.UNIQUE.beta_RMSE      = beta_RMSE_unilasso;
    summary.UNIQUE.Q_RMSE         = Q_RMSE_unilasso;
    summary.UNIQUE.ratio_same     = sign_stats_unilasso.ratio_same;
    summary.UNIQUE.ratio_opposite = sign_stats_unilasso.ratio_opposite;
    summary.UNIQUE.n_true         = sign_stats_unilasso.n_true;
    summary.UNIQUE.comp_time      = time_unilasso;
    
    % Optional extra counts
    summary.UNIQUE.n_same         = sign_stats_unilasso.n_same;
    summary.UNIQUE.n_opposite     = sign_stats_unilasso.n_opposite;
    summary.UNIQUE.n_zero         = sign_stats_unilasso.n_zero;

    % ---------------------------------------------------------
    % QR-LASSO metrics (only if requested and available)
    % ---------------------------------------------------------
    has_qrlasso = isfield(opts, 'run_qrlasso') && opts.run_qrlasso == 1 && ...
                  isfield(results, 'beta0_qrlasso') && isfield(results, 'beta_qrlasso');

    if has_qrlasso
        beta0_qrlasso = results.beta0_qrlasso;   % K x 1 or 1 x K
        beta_qrlasso  = results.beta_qrlasso;    % p x K

        if isrow(beta0_qrlasso)
            beta0_qrlasso = beta0_qrlasso(:);
        end

        sel_mask_qrlasso = abs(beta_qrlasso) > sel_thr;

        [TPR_qrlasso, FPR_qrlasso, FDR_qrlasso, MCC_qrlasso] = ...
            compute_rates_coord(sel_mask_qrlasso, true_mask);

        beta_full_qrlasso = [beta0_qrlasso'; beta_qrlasso];
        beta_RMSE_qrlasso = sqrt(mean((beta_full_qrlasso(:) - beta_full_true(:)).^2));

        Q_qrlasso_all = ones(n,1) * beta0_qrlasso' + X * beta_qrlasso;
        Q_RMSE_qrlasso = sqrt(mean((Q_qrlasso_all(:) - Q_true_all(:)).^2));
        time_qrlasso   = results.time_qrlasso;

        sign_stats_qrlasso = sign_agreement_coord( ...
            beta_qrlasso, results.beta_univ_hat, true_mask, sel_thr);

        summary.QRLASSO = struct();
        summary.QRLASSO.TPR            = TPR_qrlasso;
        summary.QRLASSO.FPR            = FPR_qrlasso;
        summary.QRLASSO.MCC            = MCC_qrlasso;
        summary.QRLASSO.FDR            = FDR_qrlasso;
        summary.QRLASSO.beta_RMSE      = beta_RMSE_qrlasso;
        summary.QRLASSO.Q_RMSE         = Q_RMSE_qrlasso;
        summary.QRLASSO.ratio_same     = sign_stats_qrlasso.ratio_same;
        summary.QRLASSO.ratio_opposite = sign_stats_qrlasso.ratio_opposite;
        summary.QRLASSO.n_true         = sign_stats_qrlasso.n_true;
        summary.QRLASSO.comp_time      = time_qrlasso;
        
        
        % Optional extra counts
        summary.QRLASSO.n_same         = sign_stats_qrlasso.n_same;
        summary.QRLASSO.n_opposite     = sign_stats_qrlasso.n_opposite;
        summary.QRLASSO.n_zero         = sign_stats_qrlasso.n_zero;
    end
end