function f = unilasso_objective(theta_vec, Y, eta_CF, tau_grid, omega, lambda)

    [n, p, K] = size(eta_CF);

    theta0 = theta_vec(1:K);
    theta  = reshape(theta_vec(K+1:end), p, K);

    % ---- Check loss ----
    loss = 0;

    for k = 1:K
        
        tau = tau_grid(k);
        
        qk = theta0(k) + eta_CF(:,:,k) * theta(:,k);
        
        u = Y - qk;
        
        rho = u .* (tau - (u < 0));
        
        loss = loss + sum(rho);
    end

    loss = loss / n;

    % ---- UniLasso penalty" sum_j (sum_k omega_jk * theta_jk)^(1/2)  ----
    pen = 0;

    for j = 1:p
        pen = pen + sqrt( sum( omega(j,:) .* theta(j,:) ) );
    end

    f = loss + lambda * pen;

end
