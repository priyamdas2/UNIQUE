function [theta0_cand, theta_cand, x_cand, Gamma_cand, ...
          Agamma_cand, feasible] = ...
          global_jump_HnR_proposal(lb, ub, p, K, beta_hat, ...
                               A, theta0, theta, ...
                               n_hit_and_run)
% ------------------------------------------------------------
% Global feasible jump using Hit-and-Run in Gamma-space
%
% Performs n_hit_and_run random-direction updates inside the
% feasible polyhedral region:
%
%   A * Gamma >= 0
%   lb <= x <= ub
%
% INPUTS:
%   theta0_curr, theta_curr : current feasible point
%   n_hit_and_run           : number of hit-and-run steps
%
% OUTPUT:
%   New feasible candidate
% ------------------------------------------------------------

if any(lb > ub)
    error('Each component of lb must be <= corresponding component of ub.');
end
feasible = true;

Gamma_curr = build_Gamma(theta0, theta, beta_hat);
Gamma = Gamma_curr;

n = K + p*K;
d = length(Gamma_curr);

lb = lb(:);
ub = ub(:);

if length(lb) ~= n || length(ub) ~= n
    error('lb and ub must both have length K + p*K.');
end

for step = 1:n_hit_and_run

    % --------------------------------------------------------
    % Step 1: Random direction
    % --------------------------------------------------------
    direction = randn(d,1);
    direction = direction / norm(direction);

    % --------------------------------------------------------
    % Step 2: Determine feasible interval
    % --------------------------------------------------------
    lower_step = -inf;
    upper_step =  inf;

    % ---- Polyhedral constraints: A * Gamma >= 0 ----
    AG = A * Gamma;
    Ad = A * direction;

    for i = 1:length(AG)
        if abs(Ad(i)) > 1e-14
            step_bound = -AG(i) / Ad(i);

            if Ad(i) > 0
                lower_step = max(lower_step, step_bound);
            else
                upper_step = min(upper_step, step_bound);
            end
        end
    end

    % ---- Box constraints on x = Gamma(1:n) ----
    for idx = 1:n
        if abs(direction(idx)) > 1e-14
            step1 = (lb(idx) - Gamma(idx)) / direction(idx);
            step2 = (ub(idx) - Gamma(idx)) / direction(idx);

            lower_step = max(lower_step, min(step1, step2));
            upper_step = min(upper_step, max(step1, step2));
        end
    end

    % No feasible movement
    if lower_step > upper_step
        continue
    end

    % --------------------------------------------------------
    % Step 3: Sample uniformly
    % --------------------------------------------------------
    step_size = lower_step + (upper_step - lower_step) * rand;
    Gamma = Gamma + step_size * direction;
end

% ------------------------------------------------------------
% Extract theta0 and theta from Gamma
% ------------------------------------------------------------
theta0_cand = Gamma(1:K);
theta_cand  = reshape(Gamma(K+1:K+p*K), p, K);

x_cand = [theta0_cand; theta_cand(:)];

Gamma_cand  = build_Gamma(theta0_cand, theta_cand, beta_hat);
Agamma_cand = A * Gamma_cand;

% Final safety checks
if any(Agamma_cand < -1e-10)
    feasible = false;
end

if any(x_cand < lb - 1e-10) || any(x_cand > ub + 1e-10)
    feasible = false;
end

end