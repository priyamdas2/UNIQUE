function stats = sign_agreement_coord_truth(beta_hat, beta_true, true_mask, sel_thr)
% =========================================================================
% sign_agreement_coord_truth
%
% Among truly active coordinates (j,k), compare the sign of beta_hat(j,k)
% against the true sign beta_true(j,k).
%
% "same"     : selected with correct sign
% "opposite" : selected with opposite sign
% "zero"     : not selected (|beta_hat| <= sel_thr)
% =========================================================================

    if nargin < 4 || isempty(sel_thr)
        sel_thr = 0;
    end

    active_idx = find(true_mask);
    n_true = numel(active_idx);

    n_same = 0;
    n_opposite = 0;
    n_zero = 0;

    for m = 1:n_true
        idx = active_idx(m);

        b_hat  = beta_hat(idx);
        b_true = beta_true(idx);

        if abs(b_hat) <= sel_thr
            n_zero = n_zero + 1;
        else
            if sign(b_hat) == sign(b_true)
                n_same = n_same + 1;
            else
                n_opposite = n_opposite + 1;
            end
        end
    end

    stats = struct();
    stats.n_true     = n_true;
    stats.n_same     = n_same;
    stats.n_opposite = n_opposite;
    stats.n_zero     = n_zero;

    if n_true > 0
        stats.ratio_same     = n_same / n_true;
        stats.ratio_opposite = n_opposite / n_true;
    else
        stats.ratio_same     = NaN;
        stats.ratio_opposite = NaN;
    end
end