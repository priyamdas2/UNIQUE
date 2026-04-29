function Gamma = build_Gamma(theta0, theta, beta_uni_hat)
% ------------------------------------------------------------
% Builds Gamma(theta) = (theta0, theta, s(theta))
%
% theta0  : K × 1
% theta   : p × K
% beta_hat: p × K
%
% Output:
%   Gamma ∈ R^{K + pK + p(K-1)}
% ------------------------------------------------------------

[p, K] = size(theta);

d = K + p*K + p*(K-1);
Gamma = zeros(d,1);

% =========================
% theta0 block
% =========================
Gamma(1:K) = theta0;

% =========================
% theta block
% =========================
for k = 1:K
    for j = 1:p
        
        idx = K + (k-1)*p + j;
        Gamma(idx) = theta(j,k);
    end
end

% =========================
% slack block s(theta)
% =========================
for k = 2:K
    for j = 1:p
        
        delta = theta(j,k-1)*beta_uni_hat(j,k-1) ...
              - theta(j,k)*beta_uni_hat(j,k);
        
        s_val = max(0, delta);
        
        idx = K + p*K + (k-2)*p + j;
        Gamma(idx) = s_val;
    end
end

end
