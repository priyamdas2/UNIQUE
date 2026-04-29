function stats = sign_agreement_coord(B_est, B_ref, true_mask, sel_thr)

% =========================================================
% Coordinate-level sign agreement summary on TRUE locations
%
% Threshold rule:
%   If |coef| <= sel_thr  -> treated as 0
%
% For each (j,k) where true_mask(j,k) = 1:
%   - Count total true signal locations
%   - Count same sign
%   - Count opposite sign
%   - Count estimated zeros
%
% Returns struct:
%   stats.n_true
%   stats.n_same
%   stats.n_opposite
%   stats.n_zero
%   stats.ratio_same
% =========================================================

    idx = find(true_mask);

    n_true      = 0;
    n_same      = 0;
    n_opposite  = 0;
    n_zero      = 0;

    for t = 1:length(idx)

        [j,k] = ind2sub(size(true_mask), idx(t));

        ref = B_ref(j,k);
        est = B_est(j,k);

        % ---- Apply threshold to reference ----
        if abs(ref) <= sel_thr
            sref = 0;
        else
            sref = sign(ref);
        end

        % ---- Apply threshold to estimate ----
        if abs(est) <= sel_thr
            sest = 0;
        else
            sest = sign(est);
        end

        n_true = n_true + 1;

        % ---- Classification ----
        if sest == 0
            n_zero = n_zero + 1;

        elseif sref == 0
            % If true reference is effectively zero (should rarely happen
            % since true_mask indicates signal), classify as opposite
            n_opposite = n_opposite + 1;

        elseif sest == sref
            n_same = n_same + 1;

        else
            n_opposite = n_opposite + 1;
        end
    end

    % Safety check
    if n_same + n_opposite + n_zero ~= n_true
        error('Sign count mismatch: components do not sum to total true signals.');
    end

    % Agreement ratio (same sign only)
    if n_true == 0
        ratio_same = NaN;
    else
        ratio_same = n_same / n_true;
        ratio_opposite = n_opposite / n_true;
    end

    % Return structured output
    stats = struct();
    stats.n_true      = n_true;
    stats.n_same      = n_same;
    stats.n_opposite  = n_opposite;
    stats.n_zero      = n_zero;
    stats.ratio_same  = ratio_same;
    stats.ratio_opposite  = ratio_opposite;

end
