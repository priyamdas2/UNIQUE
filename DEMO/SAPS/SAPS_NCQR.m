function [theta0_best, theta_best, f_best, history] = ...
    SAPS_NCQR(f, theta0, theta, beta0_uni_hat, beta_uni_hat, ...
    same_sign_mask, lb, ub, params)
% =======================================================================
% SAPS_NCQR
% SAPS: Stochastic Annealed Pattern Search
% Hybrid Global Adaptive Stochastic Descent (GLASD) with
% Simulated Annealing (SA) Global Exploration for NCQR-style constraints.
%
% -----------------------------------------------------------------------
% OVERVIEW
% -----------------------------------------------------------------------
% This routine minimizes an objective f(x) over parameters
%   theta0 ∈ R^K,  theta ∈ R^{p×K},
% where the stacked parameter vector is:
%   x = [theta0 ; theta(:)].
%
% Feasibility is enforced through linear constraints:
%   A * Gamma(theta) >= 0,
% where:
%   Gamma(theta) = (theta0, theta, s(theta)),
% and s(theta) are deterministic slack variables constructed using beta_hat.
%
% The algorithm combines:
%   (1) Greedy adaptive stochastic coordinate descent (GLASD local search)
%   (2) Periodic global jumps generated via Hit-and-Run sampling over the
%       feasible polyhedral region, accepted using simulated annealing
%       (Metropolis) probability with logarithmic cooling.
%
% -----------------------------------------------------------------------
% INPUTS
% -----------------------------------------------------------------------
% f               : function handle, takes x = [theta0; theta(:)] and returns scalar
% theta0          : K×1 initial vector
% theta           : p×K initial matrix
% beta0_uni_hat   : p×K array used to build constraint matrix A (Block 4)
% beta_uni_hat    : p×K array used to build Gamma(theta) (slack construction)
% same_sign_mask  : 
% lb              : lower bound vector of length K+pK
% ub              : upper bound vector of length K+pK
% params          : struct of algorithm parameters (optional)
%
% params fields (defaults provided if missing):
%   s_init     : initial step size for each signed direction
%   s_inc      : multiplicative increase after successful greedy step
%   s_dec      : multiplicative decrease after rejected/ infeasible step
%   p_inc      : multiplicative increase of direction probability on success
%   p_dec      : multiplicative decrease of direction probability on failure
%   m          : controls greedy vs global: greedy prob = 1 - 1/m
%   n_hit_and_run : number of hit-and-run steps per global jump
%   T          : maximum iterations
%   M          : stagnation window length
%   epsilon    : stopping tolerance for best-value improvement
%   feas_tol   : feasibility tolerance (A*Gamma >= -feas_tol)
%   c_log      : log-cooling constant; Temp = c_log / log(2 + num_global_jumps)
%
% -----------------------------------------------------------------------
% OUTPUTS
% -----------------------------------------------------------------------
% theta0_best : K×1 best theta0 found
% theta_best  : p×K best theta found
% f_best      : best objective value found
% history     : struct with fields:
%               .fvals   : best objective history
%               .x_best  : best stacked vector [theta0; theta(:)]
%
% -----------------------------------------------------------------------
% DEPENDENCIES (must be on MATLAB path)
% -----------------------------------------------------------------------
% build_A(p, K, beta0_hat)
% build_Gamma(theta0, theta, beta_hat)
% update_coordinate_value(type, j, k, theta_element_update, ...
%                         theta0, theta, Gamma, beta_hat, p, K)
% global_jump_HnR_proposal(lb, ub, p, K, beta_hat, A, theta0, theta, n_hit_and_run)
%
% =======================================================================

% -----------------------
% Dimensions and checks
% -----------------------
K = length(theta0);
[p, K_check] = size(theta);
if K_check ~= K
    error('theta and theta0 dimension mismatch');
end

% Stacked parameter vector x = [theta0; theta(:)]
x0 = [theta0; theta(:)];
n  = length(x0);

% -----------------------
% Default box bounds
% -----------------------
if nargin < 7 || isempty(lb)
    lb_theta0 = -100 * ones(K,1);

    lb_theta = -100 * ones(p*K,1);   % default unconstrained
    for kk = 1:K
        for jj = 1:p
            idx = (kk-1)*p + jj;
            if same_sign_mask(jj,kk)
                lb_theta(idx) = 0;   % constrained entries must stay nonnegative
            end
        end
    end

    lb = [lb_theta0; lb_theta];
end

if nargin < 8 || isempty(ub)
    ub_theta0 = 100 * ones(K,1);
    ub_theta  = 100 * ones(p*K,1);
    ub = [ub_theta0; ub_theta];
end

lb = lb(:);
ub = ub(:);

if length(lb) ~= n || length(ub) ~= n
    error('lb and ub must both have length K + p*K.');
end

if any(lb > ub)
    error('Each component of lb must be <= corresponding component of ub.');
end

if any(x0 < lb - 1e-12) || any(x0 > ub + 1e-12)
    error('Initial point x0 is outside the specified box bounds.');
end

% -----------------------
% Build constraint objects
% -----------------------
A      = build_A(p, K, beta0_uni_hat, same_sign_mask);
Gamma  = build_Gamma(theta0, theta, beta_uni_hat);
Agamma = A * Gamma;

feas_tol_init = max(1e-7, params.feas_tol);
bad_rows = find(Agamma < -feas_tol_init);
if ~isempty(bad_rows)
    n_sign = nnz(same_sign_mask);
    fprintf('Initial infeasible rows: ');
    fprintf('%d ', bad_rows);
    fprintf('\n');

    if any(bad_rows <= n_sign)
        fprintf('Some failed rows are sign-constraint rows.\n');
    end
    if any(bad_rows > n_sign)
        fprintf('Some failed rows are noncrossing rows.\n');
    end

    fprintf('Min Agamma = %.16e\n', min(Agamma));
    fprintf('feas_tol_init = %.16e\n', feas_tol_init);
    error('Initial point is infeasible: A*Gamma < 0 for some constraints.');
end

% -----------------------
% Default parameters
% -----------------------
default.s_init    = 0.1;
default.s_inc     = 2;
default.s_dec     = 2;
default.p_inc     = 2;
default.p_dec     = 2;
default.m         = 500;
default.T         = round(3000 * log(n));
default.M         = 8*n;
default.n_hit_and_run = 5;
default.epsilon   = 1e-12;
default.feas_tol  = 1e-8;
default.c_log     = 0.001 * log(n);

if nargin < 9 || isempty(params)
    params = default;
else
    fnames = fieldnames(default);
    for kk = 1:length(fnames)
        if ~isfield(params, fnames{kk})
            params.(fnames{kk}) = default.(fnames{kk});
        end
    end
end
% -----------------------
% Initialization
% -----------------------
x = x0;

% Direction probabilities and step sizes (2*n signed directions)
prob_vec = (1/(2*n))*ones(2*n,1);
prob_vec = prob_vec / sum(prob_vec);
s = params.s_init * ones(2*n,1);

% Objective
f_curr = f(x);
f_best = f_curr;
x_best = x;

% Stopping bookkeeping
fvals = f_best;
f_window = f_best * ones(params.M, 1);


num_global_jumps = 0;
% =======================
% Main loop
% =======================
for t = 1:params.T
    
    if rand < 1 - 1/params.m
        % ==========================================================
        % Greedy adaptive coordinate descent (GLASD local move)
        % ==========================================================
        
        % Choose a signed direction j in {1,...,2n}
        j  = randsample(2*n, 1, true, prob_vec);
        ii = ceil(j/2);                      % coordinate index in x
        sign_dir = 1 - 2*mod(j,2);           % +1 if odd, -1 if even
        
        % Propose local move in x
        x_proposed = x;
        x_proposed(ii) = x(ii) + sign_dir * s(j);
        
        % Enforce box constraints lb <= x <= ub
        if x_proposed(ii) < lb(ii) || x_proposed(ii) > ub(ii)
            s(j) = s(j) / params.s_dec;
            continue;
        end
        
        % Map ii to (theta0_k) or (theta_{j,k})
        if ii <= K
            type = 'theta0';
            k_idx = ii;
            j_idx = 0;
        else
            type = 'theta';
            idx = ii - K;
            k_idx = ceil(idx/p);
            j_idx = idx - (k_idx-1)*p;
        end
        
        % Update Gamma locally using helper
        [theta0_new, theta_new, Gamma_new, affected_rows] = ...
            update_coordinate_value(type, j_idx, k_idx, ...
            x_proposed(ii), ...
            theta0, theta, Gamma, ...
            beta_uni_hat,same_sign_mask, p, K);
        
        % Feasibility check only on affected rows
        Agamma_new = Agamma;
        if any(affected_rows < 1) || any(affected_rows > size(A,1))
            fprintf('\nDEBUG: invalid affected_rows detected\n');
            fprintf('max(affected_rows) = %d\n', max(affected_rows));
            fprintf('size(A,1)          = %d\n', size(A,1));
            fprintf('numel(affected_rows) = %d\n', numel(affected_rows));
            disp('affected_rows = ');
            disp(affected_rows(:)');
            error('affected_rows contains indices outside the row range of A.');
        end

        Agamma_new(affected_rows) = A(affected_rows,:) * Gamma_new;
        
        feasible = all(Agamma_new(affected_rows) >= -params.feas_tol);
        
        if feasible
            f_new = f(x_proposed);
            
            if f_new < f_curr
                % Accept improvement
                x      = x_proposed;
                theta0 = theta0_new;
                theta  = theta_new;
                Gamma  = Gamma_new;
                Agamma = Agamma_new;
                f_curr = f_new;
                
                % Adapt step size and probability
                s(j) = s(j) * params.s_inc;
                prob_vec(j) = prob_vec(j) * params.p_inc;
                prob_vec = prob_vec / sum(prob_vec);
            else
                % Reject (no improvement)
                s(j) = s(j) / params.s_dec;
                prob_vec(j) = prob_vec(j) / params.p_dec;
                prob_vec = prob_vec / sum(prob_vec);
            end
        else
            % Reject infeasible
            s(j) = s(j) / params.s_dec;   % prob_vec unchanged by design
        end
        
    else
        % ==========================================================
        % Simulated Annealing Global Jump
        % ==========================================================
        
        % Hit-and-Run global proposal from current feasible state
        [theta0_cand, theta_cand, x_cand, Gamma_cand, ...
            Agamma_cand, feasible_jump] = ...
            global_jump_HnR_proposal(lb, ub, p, K, beta_uni_hat, A, ...
            theta0, theta, params.n_hit_and_run);
        
        % Count every SA/global attempt
        num_global_jumps = num_global_jumps + 1;
        
        % Logarithmic cooling schedule (use current temperature now)
        Temp = params.c_log / log(2 + num_global_jumps);
        Temp = max(Temp, 1e-12);
        
        if feasible_jump
            f_new = f(x_cand);
            
            % Metropolis acceptance
            if f_new <= f_curr
                accept = true;
            else
                delta = f_new - f_curr;
                accept = (rand < exp(-delta / Temp));
            end
            
            if accept
                x      = x_cand;
                theta0 = theta0_cand;
                theta  = theta_cand;
                Gamma  = Gamma_cand;
                Agamma = Agamma_cand;
                f_curr = f_new;
            end
        end
        
        
    end
    
    % -----------------------
    % Best-so-far update
    % -----------------------
    if f_curr < f_best
        f_best = f_curr;
        x_best = x;
    end
    
    % Stopping bookkeeping
    f_window = [f_window(2:end); f_best];
    fvals(end+1,1) = f_best;
    
    if t >= params.M && (f_window(1) - f_best) < params.epsilon
        break;
    end
end

% -----------------------
% Unstack best solution
% -----------------------
theta0_best = x_best(1:K);
theta_best  = reshape(x_best(K+1:end), p, K);

history.fvals  = fvals;
history.x_best = x_best;

end

