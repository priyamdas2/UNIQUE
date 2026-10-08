function stats = sign_agreement_coord(B_est, B_ref, true_mask, sel_thr)
% =====================================================================
% Agreement with estimated MARGINAL univariate coefficient signs.
%
% B_ref must contain full-sample marginal univariate QR slopes
% from the same simulation replicate and quantile grid.
%
% Evaluation is restricted to truly active predictor-quantile coordinates.
%
% Selected: abs(B_est) > sel_thr.
% Defined reference direction: abs(B_ref) > ref_tol.
%
% Denominator:
% selected truly active coordinates with a defined marginal direction.
% =====================================================================

    if nargin < 4 || isempty(sel_thr)
        sel_thr = 0;
    end

    ref_tol = 1e-10;

    if ~isequal(size(B_est), size(B_ref), size(true_mask))
        error('B_est, B_ref, and true_mask must have identical dimensions.');
    end

    if ~isscalar(sel_thr) || ~isfinite(sel_thr) || sel_thr < 0
        error('sel_thr must be a finite, nonnegative scalar.');
    end

    idx = find(true_mask);
    est = B_est(idx);
    ref = B_ref(idx);

    if any(~isfinite(est)) || any(~isfinite(ref))
        error('Estimates and marginal references must be finite.');
    end

    selected = abs(est) > sel_thr;
    ref_defined = abs(ref) > ref_tol;
    eligible = selected & ref_defined;

    n_true = numel(idx);

    n_same = sum(eligible & (sign(est) == sign(ref)));
    n_opposite = sum(eligible & (sign(est) ~= sign(ref)));

    % Truly active coordinates not selected by this method
    n_zero = sum(~selected);

    % Selected true signals with no defined marginal direction
    n_ref_zero = sum(selected & ~ref_defined);

    stats = struct();
    stats.n_true = n_true;
    stats.n_same = n_same;
    stats.n_opposite = n_opposite;
    stats.n_zero = n_zero;
    stats.n_ref_zero = n_ref_zero;

    denom_selected = n_same + n_opposite;

    if denom_selected > 0
        stats.ratio_same = n_same / denom_selected;
        stats.ratio_opposite = n_opposite / denom_selected;
    else
        stats.ratio_same = NaN;
        stats.ratio_opposite = NaN;
    end
end