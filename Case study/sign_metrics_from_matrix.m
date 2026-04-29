function [prop_match, prop_reversed] = sign_metrics_from_matrix(beta_hat_raw, marg_sign_mat, constrained_row_mask, sel_thr)
    % beta_hat_raw: p x K, on internal/raw [0,1]-X scale
    % marg_sign_mat: p x K, training marginal sign matrix
    % constrained_row_mask: p x 1 logical
    % sel_thr: selection threshold on |beta_hat_raw|

    constrained_mask_full = repmat(constrained_row_mask(:), 1, size(beta_hat_raw,2));

    % Method-specific selected set among constrained entries only
    selected_mask = constrained_mask_full & (abs(beta_hat_raw) > sel_thr);

    denom = sum(selected_mask(:));

    if denom == 0
        prop_match = NaN;
        prop_reversed = NaN;
        return;
    end

    fitted_sign_mat = sign(beta_hat_raw);

    prop_match = sum(fitted_sign_mat(selected_mask) == marg_sign_mat(selected_mask)) / denom;
    prop_reversed = sum(fitted_sign_mat(selected_mask) == -marg_sign_mat(selected_mask)) / denom;
end
