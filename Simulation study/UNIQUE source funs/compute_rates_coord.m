function [TPR, FPR, FDR, MCC] = compute_rates_coord(sel_mask, true_mask)
% ========================================================================
% compute_rates_coord
%
% Compute coordinate-level selection metrics on the flattened (j,k) grid.
%
% INPUTS
%   sel_mask  : logical matrix, estimated selected coordinates
%   true_mask : logical matrix, true active coordinates
%
% OUTPUTS
%   TPR : true positive rate  = TP / (TP + FN)
%   FPR : false positive rate = FP / (FP + TN)
%   FDR : false discovery rate = FP / (TP + FP)
%   MCC : Matthews correlation coefficient
% ========================================================================

    sel = sel_mask(:);
    tru = true_mask(:);

    TP = sum(sel &  tru);
    FP = sum(sel & ~tru);
    TN = sum(~sel & ~tru);
    FN = sum(~sel &  tru);

    % True positive rate
    TPR = TP / max(1, TP + FN);

    % False positive rate
    FPR = FP / max(1, FP + TN);

    % False discovery rate
    FDR = FP / max(1, TP + FP);

    % Matthews correlation coefficient
    denom = sqrt((TP+FP) * (TP+FN) * (TN+FP) * (TN+FN));
    if denom < 1e-12
        MCC = 0;
    else
        MCC = (TP*TN - FP*FN) / denom;
    end
end