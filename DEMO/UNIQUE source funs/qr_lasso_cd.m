function beta = qr_lasso_cd(X, Y, tau, lambda)

[n,p] = size(X);

beta = zeros(p,1);
max_iter = 200;
tol = 1e-6;

for iter = 1:max_iter
    
    beta_old = beta;
    
    for j = 1:p
        
        r = Y - X*beta + X(:,j)*beta(j);
        
        % Soft threshold update (subgradient approximation)
        grad = -sum(X(:,j) .* (tau - (r < 0)));
        
        z = sum(X(:,j).^2);
        
        beta(j) = soft_threshold(grad/z, lambda/z);
        
    end
    
    if norm(beta - beta_old) < tol
        break;
    end
    
end

end
