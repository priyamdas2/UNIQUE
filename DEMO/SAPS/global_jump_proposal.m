function [theta0_cand, theta_cand, x_cand, Gamma_cand, ...
          Agamma_cand, feasible] = ...
          global_jump_proposal(bound, p, K, beta_hat, ...
                               A, max_trials)
% ------------------------------------------------------------
% Global feasible jump proposal inside bounded box
%
% Samples:
%   theta0_k  ~ Unif(-bound, bound)
%   theta_jk  ~ Unif(0, bound)
%
% Builds:
%   x_cand = [theta0_cand ; theta_cand(:)]
%   Gamma(theta_cand)
%   Checks A * Gamma >= 0
%
% Resamples until feasible or max_trials reached.
%
% Outputs:
%   theta0_cand  : K x 1
%   theta_cand   : p x K
%   x_cand       : stacked vector
%   Gamma_cand
%   Agamma_cand
%   feasible     : logical flag
% ------------------------------------------------------------

feasible = false;

for trial = 1:max_trials

    % --------------------------------------------------------
    % Step 1: Sample from box
    % --------------------------------------------------------

    % theta0 ~ Unif(-bound, bound)
    theta0_cand = -bound + 2*bound*rand(K,1);

    % theta >= 0 automatically
    theta_cand = bound * rand(p,K);

    % --------------------------------------------------------
    % Step 2: Build stacked x
    % --------------------------------------------------------
    x_cand = [theta0_cand ; theta_cand(:)];

    % --------------------------------------------------------
    % Step 3: Build Gamma
    % --------------------------------------------------------
    Gamma_cand = build_Gamma(theta0_cand, theta_cand, beta_hat);

    % --------------------------------------------------------
    % Step 4: Feasibility check
    % --------------------------------------------------------
    Agamma_cand = A * Gamma_cand;

    if all(Agamma_cand >= 0)
        feasible = true;
        return
    end

end

% If we exit loop, no feasible point found
theta0_cand = [];
theta_cand  = [];
x_cand      = [];
Gamma_cand  = [];
Agamma_cand = [];

end
