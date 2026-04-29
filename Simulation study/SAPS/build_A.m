function A = build_A(p, K, beta0_uni_hat, same_sign_mask)
% ------------------------------------------------------------
% Builds reduced constraint matrix A for A*Gamma >= 0
% Includes:
%   Block (1): selected theta(j,k) >= 0 constraints
%   Block (2): main global noncrossing constraints
%
% Inputs:
%   same_sign_mask : p x K logical
%
% Dimensions:
%   Gamma ∈ R^{K + pK + p(K-1)}
% ------------------------------------------------------------

    if nargin < 4 || isempty(same_sign_mask)
        same_sign_mask = true(p, K);
    end

    if ~isequal(size(same_sign_mask), [p, K])
        error('same_sign_mask must be p x K.');
    end

    same_sign_mask = logical(same_sign_mask);

    d = K + p*K + p*(K-1);
    n_sign = nnz(same_sign_mask);
    m = n_sign + (K-1);

    A = sparse(m, d);

    % =========================
    % Block (1): selected theta >= 0
    % =========================
    row_counter = 0;

    for k = 1:K
        for j = 1:p
            if same_sign_mask(j,k)
                row_counter = row_counter + 1;
                col_theta = K + (k-1)*p + j;
                A(row_counter, col_theta) = 1;
            end
        end
    end

    % =========================
    % Block (2): Noncrossing
    % =========================
    for k = 2:K

        row = n_sign + (k-1);

        % ---- theta0 part ----
        A(row, k)   =  1;
        A(row, k-1) = -1;

        % ---- theta part ----
        for j = 1:p
            col_k   = K + (k-1)*p + j;
            col_km1 = K + (k-2)*p + j;

            A(row, col_k)   = A(row, col_k)   + beta0_uni_hat(j,k);
            A(row, col_km1) = A(row, col_km1) - beta0_uni_hat(j,k-1);
        end

        % ---- slack part ----
        for j = 1:p
            col_s = K + p*K + (k-2)*p + j;
            A(row, col_s) = -1;
        end
    end
end