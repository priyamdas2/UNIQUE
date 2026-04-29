function [theta0, theta, Gamma_updated, affected_rows] = ...
    update_coordinate_value(type, j, k, theta_element_update, ...
                            theta0, theta, Gamma, beta_hat,same_sign_mask, p, K)

% ------------------------------------------------------------
% Updates one coordinate to a NEW VALUE
% Ensures consistency and numerical stability
% ------------------------------------------------------------

Gamma_updated = Gamma;
affected_rows = [];

tol = 1e-12;

% ============================================================
% CASE 1: Update theta0_k
% ============================================================
if strcmp(type,'theta0')
    
    theta0(k) = theta_element_update;
    Gamma_updated(k) = theta0(k);
    
    % Affected Block(4) rows
    if k >= 2
        affected_rows(end+1) = p*K + (k-1);
    end
    if k <= K-1
        affected_rows(end+1) = p*K + k;
    end
    
    return
end

% ============================================================
% CASE 2: Update theta_{j,k}
% ============================================================

% Enforce positivity ONLY when sign constraint is active
if same_sign_mask(j,k) && theta_element_update < 0
    error('theta_{j,k} must remain nonnegative for constrained entries');
end

theta(j,k) = theta_element_update;

% ---- Update Gamma theta block ----
idx_theta = K + (k-1)*p + j;
Gamma_updated(idx_theta) = theta(j,k);

% ---- Update slack s_{k,j} ----
if k >= 2
    
    delta_s = theta(j,k-1)*beta_hat(j,k-1) ...
            - theta(j,k)*beta_hat(j,k);
    
    if delta_s > tol
        s_val = delta_s;
    else
        s_val = 0;
    end
    
    idx_s = K + p*K + (k-2)*p + j;
    Gamma_updated(idx_s) = s_val;
end

% ---- Update slack s_{k+1,j} ----
if k <= K-1
    
    delta_s = theta(j,k)*beta_hat(j,k) ...
            - theta(j,k+1)*beta_hat(j,k+1);
    
    if delta_s > tol
        s_val = delta_s;
    else
        s_val = 0;
    end
    
    idx_s = K + p*K + (k-1)*p + j;
    Gamma_updated(idx_s) = s_val;
end

% ---- Determine affected rows ----

% Block (1)
row_block1 = (k-1)*p + j;
affected_rows(end+1) = row_block1;

% Block (4)
if k >= 2
    affected_rows(end+1) = p*K + (k-1);
end
if k <= K-1
    affected_rows(end+1) = p*K + k;
end

end

% Agamma_affected_updated = A(affected_rows,:) * Gamma_updated;
% is_feasible = all(Agamma_affected_updated >= 0);
