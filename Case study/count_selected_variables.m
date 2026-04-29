function nsel = count_selected_variables(beta_hat_raw, sel_thr)
    mean_abs_beta = mean(abs(beta_hat_raw), 2);   % p x 1
    nsel = sum(mean_abs_beta > sel_thr);
end