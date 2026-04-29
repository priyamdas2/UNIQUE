function mcc = mcc_from_sel(sel, true_sel)
    TP = sum(sel & true_sel);
    FP = sum(sel & ~true_sel);
    TN = sum(~sel & ~true_sel);
    FN = sum(~sel & true_sel);

    denom = sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN));
    if denom < 1e-12
        mcc = 0;
    else
        mcc = (TP*TN - FP*FN) / denom;
    end
end