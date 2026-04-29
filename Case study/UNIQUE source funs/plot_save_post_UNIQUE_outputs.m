function plot_save_post_UNIQUE_outputs(results, opts, outdir, prefix, design_type, plot_upto_p)
% =========================================================================
% plot_save_post_UNIQUE_outputs
%
% Creates and saves post-computation plots after running UNIQUE.
%
% Activated externally only if:
%   do_post_computation_plot == 1
%
% QR-LASSO overlays are added only if:
%   opts.run_qrlasso == 1
%
% INPUTS
%   results     : output struct from UNIQUE(...)
%   opts        : options struct used in UNIQUE
%   outdir      : directory to save figures
%   prefix      : filename prefix
%   design_type : string label for title
%   plot_upto_p : number of beta_j curves to plot
%
% OUTPUT
%   none
% =========================================================================

if ~exist(outdir, 'dir')
    mkdir(outdir);
end

% ---------------------------------------------------------
% Extract essentials
% Use original-scale X if UNIQUE internally scaled X
% ---------------------------------------------------------
if isfield(results, 'X_original')
    X = results.X_original;
else
    X = results.X;
end
tau_grid = results.tau_grid;

[n, p] = size(X);
K = numel(tau_grid);

beta0_hat_unilasso = results.beta0_hat_unilasso;
beta_hat_unilasso  = results.beta_hat_unilasso;

has_truth = isfield(opts, 'beta0_true_grid') && isfield(opts, 'beta_true_grid');
has_qrlasso = isfield(opts, 'run_qrlasso') && opts.run_qrlasso == 1 ...
    && isfield(results, 'beta0_qrlasso') && isfield(results, 'beta_qrlasso');

if has_truth
    beta0_true_grid = opts.beta0_true_grid(:);
    beta_true_grid  = opts.beta_true_grid;
end

if has_qrlasso
    beta0_qrlasso = results.beta0_qrlasso;
    if isrow(beta0_qrlasso)
        beta0_qrlasso = beta0_qrlasso(:);
    end
    beta_qrlasso  = results.beta_qrlasso;
end

% ---------------------------------------------------------
% Guard for plot_upto_p
% ---------------------------------------------------------
if nargin < 6 || isempty(plot_upto_p)
    plot_upto_p = min(p, 7);
end

if plot_upto_p < 1 || plot_upto_p > p || floor(plot_upto_p) ~= plot_upto_p
    error('plot_upto_p must be an integer between 1 and p.');
end

% =========================================================
% 1) PLOT: Quantile curve at x0 = mean(X)
% =========================================================
x0 = mean(X, 1);

Q_uni = beta0_hat_unilasso' + x0 * beta_hat_unilasso;

if has_truth
    Q_true = beta0_true_grid' + x0 * beta_true_grid;
end


if has_qrlasso
    Q_qrlasso = beta0_qrlasso' + x0 * beta_qrlasso;
end

figure('Color', 'w');

if has_truth
    plot(tau_grid, Q_true, '-o', 'LineWidth', 2, 'MarkerSize', 7); hold on;
else
    hold on;
end

plot(tau_grid, Q_uni, '-s', 'LineWidth', 2, 'MarkerSize', 7);

if has_qrlasso
    plot(tau_grid, Q_qrlasso, '-^', 'LineWidth', 2, 'MarkerSize', 7);
end

xlabel('\tau', 'FontSize', 13);
ylabel('Quantile value', 'FontSize', 13);

if has_truth && has_qrlasso
    legend({'True', 'UNIQUE', 'QR-LASSO'}, 'Location', 'NorthWest', 'FontSize', 11);
elseif has_truth && ~has_qrlasso
    legend({'True', 'UNIQUE'}, 'Location', 'NorthWest', 'FontSize', 11);
elseif ~has_truth && has_qrlasso
    legend({'UNIQUE', 'QR-LASSO'}, 'Location', 'NorthWest', 'FontSize', 11);
else
    legend({'UNIQUE'}, 'Location', 'NorthWest', 'FontSize', 11);
end

title(sprintf('Quantile Curves at Mean(X) (n=%d, p=%d, design=%s)', ...
    n, p, design_type));
set(gca, 'FontSize', 12);
grid on;
box on;

fname = fullfile(outdir, sprintf('%s_quantile_curves_at_mean.png', prefix));
exportgraphics(gcf, fname, 'Resolution', 300);

% =========================================================
% 2) PLOT: True vs Estimated Quantiles over all observations
%    Only if truth is available
% =========================================================
if has_truth
    Q_true_all     = ones(n,1) * beta0_true_grid'     + X * beta_true_grid;
    Q_unilasso_all = ones(n,1) * beta0_hat_unilasso'  + X * beta_hat_unilasso;
    
    q_true_vec = Q_true_all(:);
    q_uni_vec  = Q_unilasso_all(:);
    
    if has_qrlasso
        Q_qrlasso_all = ones(n,1) * beta0_qrlasso' + X * beta_qrlasso;
        q_qrlasso_vec = Q_qrlasso_all(:);
        qmin = min([q_true_vec; q_uni_vec; q_qrlasso_vec]);
        qmax = max([q_true_vec; q_uni_vec; q_qrlasso_vec]);
    else
        qmin = min([q_true_vec; q_uni_vec]);
        qmax = max([q_true_vec; q_uni_vec]);
    end
    
    colors = lines(K);
    
    % ---------- UNIQUE ----------
    figure('Color', 'w');
    hold on;
    
    for k = 1:K
        scatter(Q_true_all(:,k), Q_unilasso_all(:,k), 18, ...
            'MarkerFaceColor', colors(k,:), ...
            'MarkerEdgeColor', 'none', ...
            'MarkerFaceAlpha', 0.35);
    end
    
    plot([qmin qmax], [qmin qmax], 'k--', 'LineWidth', 2);
    
    xlabel('True quantile value', 'FontSize', 13);
    ylabel('Estimated quantile value', 'FontSize', 13);
    title(sprintf('True vs Estimated Quantiles (UNIQUE) (n=%d, p=%d, design=%s)', ...
        n, p, design_type));
    
    set(gca, 'FontSize', 12);
    grid on; box on;
    axis equal;
    xlim([qmin qmax]);
    ylim([qmin qmax]);
    
    legend(arrayfun(@(t) sprintf('\\tau=%.1f', t), tau_grid, 'UniformOutput', false), ...
        'Location', 'bestoutside');
    
    fname = fullfile(outdir, sprintf('%s_true_vs_est_all_points_UNIQUE.png', prefix));
    exportgraphics(gcf, fname, 'Resolution', 300);
    
    % ---------- QR-LASSO ----------
    if has_qrlasso
        figure('Color', 'w');
        hold on;
        
        for k = 1:K
            scatter(Q_true_all(:,k), Q_qrlasso_all(:,k), 18, ...
                'MarkerFaceColor', colors(k,:), ...
                'MarkerEdgeColor', 'none', ...
                'MarkerFaceAlpha', 0.35);
        end
        
        plot([qmin qmax], [qmin qmax], 'k--', 'LineWidth', 2);
        
        xlabel('True quantile value', 'FontSize', 13);
        ylabel('Estimated quantile value', 'FontSize', 13);
        title(sprintf('True vs Estimated Quantiles (QR-LASSO) (n=%d, p=%d, design=%s)', ...
            n, p, design_type));
        
        set(gca, 'FontSize', 12);
        grid on; box on;
        axis equal;
        xlim([qmin qmax]);
        ylim([qmin qmax]);
        
        legend(arrayfun(@(t) sprintf('\\tau=%.1f', t), tau_grid, 'UniformOutput', false), ...
            'Location', 'bestoutside');
        
        fname = fullfile(outdir, sprintf('%s_true_vs_est_all_points_QRLASSO.png', prefix));
        exportgraphics(gcf, fname, 'Resolution', 300);
    end
end

% =========================================================
% 3) PLOT: Coefficient curves
% =========================================================
n_panels = plot_upto_p + 1;
ncol = 2;
nrow = ceil(n_panels / ncol);

figure('Color', 'w');

% ---------- beta_0 ----------
subplot(nrow, ncol, 1);
hold on;

if has_truth
    plot(tau_grid, beta0_true_grid, '-o', 'LineWidth', 2, 'MarkerSize', 6);
end

plot(tau_grid, beta0_hat_unilasso, '-s', 'LineWidth', 2, 'MarkerSize', 6);

if has_qrlasso
    plot(tau_grid, beta0_qrlasso, '-^', 'LineWidth', 2, 'MarkerSize', 6);
end

xlabel('\tau', 'FontSize', 12);
ylabel('\beta_0(\tau)', 'FontSize', 12);
title('\beta_0(\tau)');

if has_truth && has_qrlasso
    legend({'True', 'UNIQUE', 'QR-LASSO'}, 'Location', 'best', 'FontSize', 10);
elseif has_truth && ~has_qrlasso
    legend({'True', 'UNIQUE'}, 'Location', 'best', 'FontSize', 10);
elseif ~has_truth && has_qrlasso
    legend({'UNIQUE', 'QR-LASSO'}, 'Location', 'best', 'FontSize', 10);
else
    legend({'UNIQUE'}, 'Location', 'best', 'FontSize', 10);
end

grid on;
box on;
set(gca, 'FontSize', 11);

% ---------- beta_1 to beta_{plot_upto_p} ----------
for j = 1:plot_upto_p
    subplot(nrow, ncol, j+1);
    hold on;
    
    if has_truth
        beta_true_plot = beta_true_grid(j,:);
        
        if all(abs(beta_true_plot) < 1e-14)
            title_str = sprintf('\\beta_{%d}(\\tau) [true = 0]', j);
        else
            title_str = sprintf('\\beta_{%d}(\\tau)', j);
        end
        
        plot(tau_grid, beta_true_plot, '-o', 'LineWidth', 2, 'MarkerSize', 6);
    else
        title_str = sprintf('\\beta_{%d}(\\tau)', j);
    end
    
    plot(tau_grid, beta_hat_unilasso(j,:), '-s', 'LineWidth', 2, 'MarkerSize', 6);
    
    if has_qrlasso
        plot(tau_grid, beta_qrlasso(j,:), '-^', 'LineWidth', 2, 'MarkerSize', 6);
    end
    
    xlabel('\tau', 'FontSize', 12);
    ylabel(sprintf('\\beta_{%d}(\\tau)', j), 'FontSize', 12);
    title(title_str);
    grid on;
    box on;
    set(gca, 'FontSize', 11);
end

sgtitle(sprintf('Coefficient Curves (n=%d, p=%d, design=%s)', ...
    n, p, design_type), 'FontSize', 14);

fname = fullfile(outdir, sprintf('%s_beta_curves_upto_p_%d.png', prefix, plot_upto_p));
exportgraphics(gcf, fname, 'Resolution', 300);

% =========================================================
% 4) OPTIONAL PLOT: Lambda vs CV path
% =========================================================
if isfield(results, 'cv_loss_path') && isfield(results, 'lambda_opt') && isfield(results, 'idx_opt')
    figure('Color', 'w');
    
    semilogx(opts.lambda_grid, results.cv_loss_path, '-o', ...
        'LineWidth', 2, 'MarkerSize', 6);
    hold on;
    
    plot(results.lambda_opt, results.cv_loss_path(results.idx_opt), 'rp', ...
        'MarkerSize', 12, 'MarkerFaceColor', 'r');
    
    xlabel('\lambda', 'FontSize', 13);
    ylabel('Check loss (test)', 'FontSize', 13);
    title(sprintf('UNIQUE Lambda vs CV Check Loss (n=%d, p=%d, design=%s)', ...
        n, p, design_type));
    
    grid on;
    box on;
    set(gca, 'FontSize', 12);
    legend({'CV loss', 'Selected \lambda'}, 'Location', 'best');
    
    fname = fullfile(outdir, sprintf('%s_lambda_vs_cv.png', prefix));
    exportgraphics(gcf, fname, 'Resolution', 300);
end

fprintf('\nPost-computation plots saved in folder: %s\n', outdir);
end