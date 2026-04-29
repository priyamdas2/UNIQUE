function A = build_A(p, K, beta0_uni_hat, same_sign_mask)
% ------------------------------------------------------------
% Builds constraint matrix A for A*Gamma >= 0
%
% IMPORTANT:
% We preserve the ORIGINAL row layout so that helper functions
% such as update_coordinate_value() continue to work unchanged.
%
% Row layout:
%   Block (1): rows 1 : p*K
%              theta(j,k) >= 0 constraints
%              If same_sign_mask(j,k) = false, the row is left all-zero.
%
%   Block (2): rows p*K + 1 : p*K + (K-1)
%              global noncrossing constraints
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

    % --------------------------------------------------------
    % Keep full row layout:
    %   p*K sign rows + (K-1) noncrossing rows
    % --------------------------------------------------------
    m = p*K + (K-1);

    A = sparse(m, d);

    % =========================
    % Block (1): theta(j,k) >= 0
    % Only activate rows where same_sign_mask(j,k) = true.
    % Unconstrained rows remain all-zero.
    % =========================
    for k = 1:K
        for j = 1:p
            row = (k-1)*p + j;
            if same_sign_mask(j,k)
                col_theta = K + (k-1)*p + j;
                A(row, col_theta) = 1;
            end
        end
    end

    % =========================
    % Block (2): Noncrossing
    % Row indexing preserved to match update_coordinate_value():
    %   row = p*K + (k-1),  for k = 2,...,K
    % =========================
    for k = 2:K

        row = p*K + (k-1);

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