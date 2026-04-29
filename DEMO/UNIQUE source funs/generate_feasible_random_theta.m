function [theta0_init, theta_init, attempts_made] = ...
    generate_feasible_random_theta(p, K, beta0_hat, beta_hat, ...
                                   same_sign_mask, lb, ub, max_attempts)
% =========================================================
% Generate random feasible starting point for UNIQUE
%
% Strategy:
%   1. Randomly draw theta within box bounds
%      - constrained entries: theta(j,k) >= 0
%      - unconstrained entries: theta(j,k) can be negative
%   2. Randomly draw theta0 within box bounds
%   3. Repair noncrossing via push-up on theta0
%   4. Check feasibility
%   5. Repeat until feasible (max_attempts)
%
% INPUTS
%   p, K            : dimensions
%   beta0_hat       : p x K array used in build_A
%   beta_hat        : p x K array used in build_Gamma
%   same_sign_mask  : p x K logical, true => theta(j,k) >= 0
%   lb, ub          : bounds for x = [theta0; theta(:)]
%   max_attempts    : max number of attempts
%
% OUTPUTS
%   theta0_init, theta_init, attempts_made
% =========================================================

    if nargin < 8 || isempty(max_attempts)
        max_attempts = 50;
    end

    if nargin < 5 || isempty(same_sign_mask)
        same_sign_mask = true(p, K);
    end

    if ~isequal(size(same_sign_mask), [p, K])
        error('same_sign_mask must be p x K.');
    end
    same_sign_mask = logical(same_sign_mask);

    npar = K + p*K;

    if nargin < 6 || isempty(lb)
        lb_theta0 = -100 * ones(K,1);
        lb_theta_mat = -100 * ones(p, K);
        lb_theta_mat(same_sign_mask) = 0;
        lb = [lb_theta0; lb_theta_mat(:)];
    end

    if nargin < 7 || isempty(ub)
        ub = 100 * ones(npar,1);
    end

    lb = lb(:);
    ub = ub(:);

    if length(lb) ~= npar || length(ub) ~= npar
        error('lb and ub must both have length K + p*K.');
    end

    if any(lb > ub)
        error('Each component of lb must be <= corresponding component of ub.');
    end

    margin   = 1e-8;
    feas_tol = 0;

    A0 = build_A(p, K, beta0_hat, same_sign_mask);

    for attempt = 1:max_attempts

        % --------------------------
        % 1. Random slopes within bounds
        % --------------------------
        theta_init = zeros(p, K);

        for k = 1:K
            for j = 1:p
                idx = K + (k-1)*p + j;

                if same_sign_mask(j,k)
                    % Positive-only draw within [lb, ub], typically [0, 100]
                    theta_init(j,k) = lb(idx) + (ub(idx) - lb(idx)) * rand;
                else
                    % Free-sign draw within [lb, ub], typically [-100, 100]
                    theta_init(j,k) = lb(idx) + (ub(idx) - lb(idx)) * rand;
                end
            end
        end

        % Optional normalization for stability
        max_abs_theta = max(abs(theta_init(:)));
        if max_abs_theta > 1
            theta_init = theta_init / max_abs_theta;
        end

        % Re-clip after normalization to respect bounds
        for k = 1:K
            for j = 1:p
                idx = K + (k-1)*p + j;
                theta_init(j,k) = min(max(theta_init(j,k), lb(idx)), ub(idx));
            end
        end

        % --------------------------
        % 2. Random intercepts within bounds
        % --------------------------
        theta0_init = zeros(K,1);
        for k = 1:K
            theta0_init(k) = lb(k) + (ub(k) - lb(k)) * rand;
        end

        % --------------------------
        % 3. Enforce global noncrossing via push-up
        % --------------------------
        for k = 2:K

            s_k = max(0, ...
                theta_init(:,k-1).*beta_hat(:,k-1) - ...
                theta_init(:,k).*beta_hat(:,k));

            drift = sum(theta_init(:,k).*beta0_hat(:,k) - ...
                        theta_init(:,k-1).*beta0_hat(:,k-1));

            rhs = theta0_init(k-1) - drift + sum(s_k) + margin;

            if theta0_init(k) < rhs
                theta0_init(k) = rhs;
            end

            % Respect upper bound on theta0(k)
            theta0_init(k) = min(theta0_init(k), ub(k));
        end

        % --------------------------
        % 4. Hard feasibility check
        % --------------------------
        x0 = [theta0_init; theta_init(:)];

        % Box feasibility
        if any(x0 < lb - feas_tol) || any(x0 > ub + feas_tol)
            attempts_made = attempt;
            continue;
        end

        Gamma0 = build_Gamma(theta0_init, theta_init, beta_hat);
        Ag0    = A0 * Gamma0;

        attempts_made = attempt;

        if min(Ag0) >= -feas_tol
            fprintf('Feasible random initialization found at attempt %d\n', attempt);
            return;
        end
    end

    error('Failed to generate feasible random initialization after %d attempts.', max_attempts);
end