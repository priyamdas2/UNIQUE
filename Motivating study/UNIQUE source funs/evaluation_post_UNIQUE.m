function metrics = evaluation_post_UNIQUE(results, opts)

    % ---------------------------------------------------------
    % Use original X scale if available
    % ---------------------------------------------------------
    if isfield(results, 'X_original')
        X = results.X_original;
    else
        X = results.X;
    end

    tau_grid = results.tau_grid;

    beta0_true_grid = opts.beta0_true_grid(:);
    beta_true_grid  = opts.beta_true_grid;
    sel_thr         = opts.sel_thr;

    beta0_hat_unilasso = results.beta0_hat_unilasso;
    beta_hat_unilasso  = results.beta_hat_unilasso;

    true_support = opts.true_support;   % now expected to be p x K matrix
    [n, p] = size(X);
    K = numel(tau_grid);

    % ---------------------------------------------------------
    % TRUE mask: p x K logical matrix
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

    beta_full_unilasso = [beta0_hat_unilasso'; beta_hat_unilasso];
    beta_full_true     = [beta0_true_grid'; beta_true_grid];

    beta_rmse_unilasso = sqrt(mean((beta_full_unilasso(:) - beta_full_true(:)).^2));

    Q_true_all     = ones(n,1) * beta0_true_grid'    + X * beta_true_grid;
    Q_unilasso_all = ones(n,1) * beta0_hat_unilasso' + X * beta_hat_unilasso;

    Q_rmse_unilasso = sqrt(mean((Q_unilasso_all(:) - Q_true_all(:)).^2));
    Q_mad_unilasso  = mean(abs(Q_unilasso_all(:) - Q_true_all(:)));

    stats_unilasso = sign_agreement_coord( ...
        beta_hat_unilasso, results.beta_univ_hat, true_mask, sel_thr);

    metrics = struct();

    metrics.TPR_unilasso = TPR_unilasso;
    metrics.FPR_unilasso = FPR_unilasso;
    metrics.FDR_unilasso = FDR_unilasso;
    metrics.MCC_unilasso = MCC_unilasso;

    metrics.beta_rmse_unilasso = beta_rmse_unilasso;
    metrics.Q_rmse_unilasso    = Q_rmse_unilasso;
    metrics.Q_mad_unilasso     = Q_mad_unilasso;
    metrics.sign_stats_unilasso = stats_unilasso;

    % =========================================================
    % QR-LASSO metrics (if available)
    % =========================================================
    if opts.run_qrlasso == 1
        beta0_qrlasso = results.beta0_qrlasso;
        beta_qrlasso  = results.beta_qrlasso;

        if isrow(beta0_qrlasso)
            beta0_qrlasso = beta0_qrlasso(:);
        end

        sel_mask_qrlasso = abs(beta_qrlasso) > sel_thr;

        [TPR_qrlasso, FPR_qrlasso, FDR_qrlasso, MCC_qrlasso] = ...
            compute_rates_coord(sel_mask_qrlasso, true_mask);

        beta_full_qrlasso = [beta0_qrlasso'; beta_qrlasso];
        beta_rmse_qrlasso = sqrt(mean((beta_full_qrlasso(:) - beta_full_true(:)).^2));

        Q_qrlasso_all = ones(n,1) * beta0_qrlasso' + X * beta_qrlasso;
        Q_rmse_qrlasso = sqrt(mean((Q_qrlasso_all(:) - Q_true_all(:)).^2));
        Q_mad_qrlasso  = mean(abs(Q_qrlasso_all(:) - Q_true_all(:)));

        stats_qrlasso = sign_agreement_coord( ...
            beta_qrlasso, results.beta_univ_hat, true_mask, sel_thr);

        metrics.TPR_qrlasso = TPR_qrlasso;
        metrics.FPR_qrlasso = FPR_qrlasso;
        metrics.FDR_qrlasso = FDR_qrlasso;
        metrics.MCC_qrlasso = MCC_qrlasso;

        metrics.beta_rmse_qrlasso = beta_rmse_qrlasso;
        metrics.Q_rmse_qrlasso    = Q_rmse_qrlasso;
        metrics.Q_mad_qrlasso     = Q_mad_qrlasso;
        metrics.sign_stats_qrlasso = stats_qrlasso;
    end

end