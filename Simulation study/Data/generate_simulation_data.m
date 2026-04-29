function data = generate_simulation_data(n, p, p_true, tau_grid, design_type, do_plot)
% ========================================================================
% generate_simulation_data
%
% Generate data from a quantile-varying sparse linear model:
%
%   Y = beta0(U) + sum_{j=1}^{p_true} X_j * beta_j(U),   U ~ Unif(0,1)
%
% so that the true conditional quantile function is
%
%   Q_Y(tau | X) = beta0(tau) + sum_{j=1}^{p_true} X_j * beta_j(tau),
%
% with beta_j(tau) = 0 for j > p_true.
%
% ------------------------------------------------------------------------
% INPUTS
% ------------------------------------------------------------------------
% n           : sample size
% p           : total number of covariates (must satisfy p >= p_true)
% p_true      : number of truly active predictors, must be in {1,2,3,4,5}
% tau_grid    : vector of quantile levels in (0,1), e.g. 0.1:0.1:0.9
% design_type : 'indep' or 'corr'
%               'indep' : X_ij ~ Uniform(0,1), independent across j
%               'corr'  : Gaussian copula with AR(1)-type correlation,
%                         then transformed to Uniform(0,1) margins
% do_plot     : logical, whether to plot beta_0,...,beta_5
%               default = true
%
% ------------------------------------------------------------------------
% TRUE COEFFICIENT FUNCTIONS
% ------------------------------------------------------------------------
% beta0(tau): starts at -4, ends at 4, symmetric tanh-type curve
% beta1(tau): starts at 2, ends at 7, quadratic increasing
% beta2(tau): linearly decreases from -4 to -6
% beta3(tau): increase 2 to 6 (at tau = 0.5) then constant at 6
% beta4(tau): constant at 5
%
% If p_true < 4, only the first p_true of beta1,...,beta4 are used.
% Remaining predictors are noise with coefficient identically zero.
%
% ------------------------------------------------------------------------
% OUTPUT (struct)
% ------------------------------------------------------------------------
% data.X               : n x p design matrix
% data.Y               : n x 1 response vector
% data.U               : n x 1 latent uniforms used in data generation
% data.tau_grid        : quantile grid
% data.beta_true_grid  : p x K matrix of true slope values on tau_grid
% data.beta0_true_grid : 1 x K vector of true intercept values on tau_grid
% data.true_support    : cell(K,1), active set at each tau in tau_grid
% data.beta_funs       : cell array of function handles {beta0,...,beta5}
% data.design_type     : copied input
% data.p_true          : copied input
%
% ------------------------------------------------------------------------
% NOTES
% ------------------------------------------------------------------------
% 1. This construction gives an exact conditional quantile model by using
%    the latent-uniform device U ~ Unif(0,1).
%
% 2. The overall quantile function is:
%       Q_Y(tau|X) = beta0(tau) + X_1 beta1(tau) + ... + X_{p_true} beta_{p_true}(tau)
%
% 3. For j > p_true, beta_j(tau) = 0 for all tau.
% ========================================================================

% ----------------------------
% Default plotting option
% ----------------------------
if nargin < 6 || isempty(do_plot)
    do_plot = true;
end

% ----------------------------
% Basic input checks
% ----------------------------
if p_true < 1 || p_true > 5 || floor(p_true) ~= p_true
    error('p_true must be an integer between 1 and 5.');
end

if p < p_true
    error('p must be at least p_true.');
end

tau_grid = tau_grid(:);
if any(tau_grid <= 0 | tau_grid >= 1)
    error('tau_grid values must lie strictly between 0 and 1.');
end

K = length(tau_grid);

% ============================================================
% 1. Generate X
% ============================================================
switch lower(design_type)
    
    case 'indep'
        % Independent Uniform(0,1) covariates
        X = rand(n, p);
    case 'corr'
        % Correlated Gaussian copula design with Uniform(0,1) margins
        rho = 0.5;
        Sigma = toeplitz(rho.^(0:p-1));
        Z = mvnrnd(zeros(1,p), Sigma, n);
        X = normcdf(Z);
    otherwise
        error('design_type must be ''indep'' or ''corr''.');
end

% ============================================================
% 2. Define true coefficient functions
% ============================================================

% beta0(tau): starts at 3, ends at 6, smooth symmetric tanh-shape
% The normalization by tanh(2) ensures exact endpoints:
%   beta0(0)=-4, beta0(1)=4
beta0_fun = @(tau) -0.0 + 4 * tanh(4*(tau - 0.5)) / tanh(2);

% beta1(tau): quadratic increase from 2 to 7
beta1_fun = @(tau) 2 + 5*tau.^2;

% beta2(tau): linear decrease from -4 to -6
beta2_fun = @(tau) -4 - 2*tau;

% beta3(tau): increase 2 to 6 (at tau = 0.5) then constant at 6
beta3_fun = @(tau) (tau <= 0.5).*(2+8*tau) + (tau > 0.5).*(6-0*(tau-0.5));

% beta4(tau): constant at 5
beta4_fun = @(tau) 5+0*tau;


% Store handles in a cell array for convenience
beta_funs = {beta1_fun, beta2_fun, beta3_fun, beta4_fun};

% ============================================================
% 3. Generate Y using latent U ~ Uniform(0,1)
% ============================================================
%
% This guarantees the true conditional quantile function:
%   Q_Y(tau | X) = beta0(tau) + sum_{j=1}^{p_true} X_j beta_j(tau).
%
U = rand(n,1);
Y = zeros(n,1);

for i = 1:n
    u = U(i);
    
    % Intercept contribution
    yi = beta0_fun(u);
    
    % Add active predictor contributions
    for j = 1:p_true
        yi = yi + X(i,j) * beta_funs{j}(u);
    end
    
    Y(i) = yi;
end

% ============================================================
% 4. Evaluate true coefficients on the supplied tau grid
% ============================================================
beta0_true_grid = beta0_fun(tau_grid).';   % 1 x K
beta_true_grid  = zeros(p, K);
for k = 1:K
    tau = tau_grid(k);

    for j = 1:p_true
        beta_true_grid(j,k) = beta_funs{j}(tau);
    end
end

% Logical support matrix
true_support = abs(beta_true_grid) > 1e-12;

% ============================================================
% Optional plot of beta_0,...,beta_5
% If p_true < 5, inactive beta_j curves are plotted as zero.
% ============================================================
if do_plot
    tau_plot = linspace(0,1,400);
    
    figure('Color','w');
    
    % ---- beta_0 ----
    subplot(3,2,1);
    plot(tau_plot, beta0_fun(tau_plot), 'LineWidth', 2);
    xlabel('\tau');
    ylabel('\beta_0(\tau)');
    title('\beta_0(\tau)');
    xlim([0 1]);
    grid on; box on;
    
    % Store active function handles
    beta_fun_list = {beta1_fun, beta2_fun, beta3_fun, beta4_fun};
    
    % ---- beta_1,...,beta_5 ----
    for j = 1:4
        subplot(3,2,j+1);
        
        if j <= p_true
            beta_plot = beta_fun_list{j}(tau_plot);
            title_str = sprintf('\\beta_{%d}(\\tau)', j);
        else
            beta_plot = zeros(size(tau_plot));
            title_str = sprintf('\\beta_{%d}(\\tau) [true = 0]', j);
        end
        
        plot(tau_plot, beta_plot, 'LineWidth', 2);
        xlabel('\tau');
        ylabel(sprintf('\\beta_{%d}(\\tau)', j));
        title(title_str);
        xlim([0 1]);
        grid on; box on;
    end
    
    sgtitle(sprintf('True coefficient functions (p_{true} = %d)', p_true));
end
% ============================================================
% 6. Return output struct
% ============================================================
data = struct();
data.X = X;
data.Y = Y;
data.U = U;
data.tau_grid = tau_grid;
data.beta_true_grid = beta_true_grid;
data.beta0_true_grid = beta0_true_grid;
data.true_support = true_support;
data.beta_funs = [{beta0_fun}, beta_funs];
data.design_type = design_type;
data.p_true = p_true;

end

function b_vec = compute_b(u, p, S_A, S_L, S_H, ...
                           c_vals, d_vals, e_vals)

% ------------------------------------------------------------
% Monotone noncrossing quantile coefficient structure
%
% Guarantees:
%   • Q(tau|x) nondecreasing for x in [0,1]^p
%   • Indicator thresholds at 0.35 and 0.75
%   • All active magnitudes > 1
%   • Smooth monotone growth within regions
% ------------------------------------------------------------

b_vec = zeros(p,1);


gamma_always = 4;
gamma_low = 4;
gamma_high = 0.2;

%% Always-active
for j = 1:length(S_A)
    idx = S_A(j);
    
    base = abs(c_vals(j)) + 1;
    sign_j = sign(c_vals(j));
    
    % signed base + positive slope
    b_vec(idx) = sign_j * base + gamma_always*u;
end

%% Low-quantile active (negative block)
if u <= 0.35
    for j = 1:length(S_L)
        idx = S_L(j);
        
        base = abs(d_vals(j)) + 1;
        
        % strictly increasing in u
        b_vec(idx) = -base + gamma_low*u;
    end
end

%% High-quantile active (positive block)
if u >= 0.75
    for j = 1:length(S_H)
        idx = S_H(j);
        
        base = abs(e_vals(j)) + 1;
        
        b_vec(idx) = base + gamma_high*u;
    end
end

end
