function [eta_CF, fold_id] = compute_grouped_crossfit_eta(X, Y, tau_grid, num_folds_for_crossfit, verbose_crossfit)
% =========================================================================
% compute_grouped_crossfit_eta
%
% Computes grouped cross-fitted univariate quantile-regression meta-features:
%   eta_CF(i,j,k) = out-of-fold prediction for observation i,
%                   predictor j, quantile tau_k
%
% INPUTS
%   X   : n x p design matrix
%   Y   : n x 1 response
%   tau_grid : 1 x K vector of quantile levels
%   num_folds_for_crossfit : number of folds
%   verbose_crossfit : 0/1
%
% OUTPUTS
%   eta_CF  : n x p x K array of cross-fitted predictions
%   fold_id : n x 1 fold assignments
% =========================================================================

    [n, p] = size(X);
    K = numel(tau_grid);

    % -----------------------------
    % Balanced fold assignment
    % -----------------------------
    base_size = floor(n / num_folds_for_crossfit);
    remainder = mod(n, num_folds_for_crossfit);

    perm = randperm(n);
    fold_id = zeros(n,1);

    start_idx = 1;
    for m = 1:num_folds_for_crossfit
        if m <= remainder
            fold_size = base_size + 1;
        else
            fold_size = base_size;
        end

        idx = perm(start_idx:start_idx + fold_size - 1);
        fold_id(idx) = m;
        start_idx = start_idx + fold_size;
    end

    % -----------------------------
    % Cross-fitted meta-features
    % -----------------------------
    eta_CF = zeros(n, p, K);

    for m = 1:num_folds_for_crossfit
        tStart = tic;

        if verbose_crossfit == 1
            fprintf('Performing cross-fit fold %d / %d... ', ...
                m, num_folds_for_crossfit);
        end

        test_idx  = (fold_id == m);
        train_idx = ~test_idx;

        Y_train = Y(train_idx);
        X_train = X(train_idx,:);
        X_test  = X(test_idx,:);

        for j = 1:p
            xj_train = X_train(:,j);

            for k = 1:K
                tau = tau_grid(k);

                % Univariate QR fit on training fold
                [b0, b1] = qr_linprog(xj_train, Y_train, tau);

                % Cross-fitted prediction on test fold
                eta_CF(test_idx,j,k) = b0 + b1 * X_test(:,j);
            end
        end

        if verbose_crossfit == 1
            fprintf('done (%.2f sec)\n', toc(tStart));
        end
    end
end