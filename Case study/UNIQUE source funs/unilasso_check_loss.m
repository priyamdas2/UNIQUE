function val = unilasso_check_loss(theta0, theta, Y, eta_CF, tau_grid)

    [n, ~, K] = size(eta_CF);
    val = 0;

    for k = 1:K
        tau = tau_grid(k);
        qk  = theta0(k) + eta_CF(:,:,k) * theta(:,k);
        u   = Y - qk;
        rho = u .* (tau - (u < 0));
        val = val + sum(rho);
    end

    val = val / n;

end