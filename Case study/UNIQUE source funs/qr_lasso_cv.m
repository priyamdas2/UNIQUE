function [beta0_hat, beta_hat, lambda_opt, cv_loss] = ...
    qr_lasso_cv(X, Y, tau_grid, lambda_grid, num_folds)
% ========================================================================
% QR_LASSO_CV
%
% K-fold CV to choose lambda_k separately for each tau_k for QR-LASSO.
% Uses a proper linprog-based penalized quantile regression solver.
%
% OUTPUTS
%   beta0_hat  : K x 1 intercepts
%   beta_hat   : p x K slopes
%   lambda_opt : 1 x K selected lambda per quantile
%   cv_loss    : K x L CV loss matrix
% ========================================================================

    [n, p] = size(X);
    K = length(tau_grid);
    L = length(lambda_grid);

    Y = Y(:);

    %% ------------------------------------------------
    % Build balanced folds
    %% ------------------------------------------------
    base_size = floor(n / num_folds);
    remainder = mod(n, num_folds);

    perm = randperm(n);
    fold_id = zeros(n,1);

    start_idx = 1;
    for m = 1:num_folds
        if m <= remainder
            fold_size = base_size + 1;
        else
            fold_size = base_size;
        end

        idx = perm(start_idx:start_idx + fold_size - 1);
        fold_id(idx) = m;
        start_idx = start_idx + fold_size;
    end

    %% ------------------------------------------------
    % Cross-validation (separate lambda per quantile)
    %% ------------------------------------------------
    cv_loss = zeros(K, L);

    for k = 1:K
        tau = tau_grid(k);

        for ell = 1:L
            lambda = lambda_grid(ell);
            loss_accum = 0;

            for m = 1:num_folds
                test_idx  = (fold_id == m);
                train_idx = ~test_idx;

                X_train = X(train_idx, :);
                Y_train = Y(train_idx);
                X_test  = X(test_idx, :);
                Y_test  = Y(test_idx);

                % Proper QR-LASSO fit with intercept included in LP
                [beta0_k, beta_k] = qr_lasso_linprog(X_train, Y_train, tau, lambda);

                % Predict on test fold
                Y_pred = beta0_k + X_test * beta_k;

                % Check loss on test fold
                u = Y_test - Y_pred;
                rho = u .* (tau - (u < 0));
                loss_accum = loss_accum + sum(rho);
            end

            cv_loss(k, ell) = loss_accum / n;
        end
    end

    %% ------------------------------------------------
    % Select optimal lambda per quantile
    %% ------------------------------------------------
    lambda_opt = zeros(1, K);

    for k = 1:K
        [~, idx_opt] = min(cv_loss(k, :));
        lambda_opt(k) = lambda_grid(idx_opt);
    end


    %% ------------------------------------------------
    % Refit on full data using lambda_k
    %% ------------------------------------------------
    beta_hat  = zeros(p, K);
    beta0_hat = zeros(K, 1);

    for k = 1:K
        tau = tau_grid(k);
        lambda_k = lambda_opt(k);

        [beta0_k, beta_k] = qr_lasso_linprog(X, Y, tau, lambda_k);

        beta0_hat(k)  = beta0_k;
        beta_hat(:,k) = beta_k;
    end
end