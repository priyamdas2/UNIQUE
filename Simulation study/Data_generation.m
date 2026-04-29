clear; clc;
addpath('./Data/');
datadir = 'Data';

%% =======================
%  DATA GENERATION
%% =======================
Num_exps = 20;
n = 500;
p = 100;
p_true = 4;
tau_grid_for_TRUE_generation = 0.1:0.1:0.9;
K = length(tau_grid_for_TRUE_generation);
design_type = 'indep'; % 'corr'/ 'indep'
do_TRUE_quantile_plot = false;

fprintf(' Generating data: n = %d, p = %d, design = %s\n', n, p, design_type);
save_beta_once_counter = 0;

for rep_num = 1:Num_exps
    rng(rep_num);

    fprintf(' Rep = %d\n', rep_num);

    % Generate data
    data = generate_simulation_data(n, p, p_true, ...
        tau_grid_for_TRUE_generation, design_type, do_TRUE_quantile_plot);

    % =====================================================
    % Save [Y, X] as n x (p+1) matrix
    % First column = Y, remaining columns = X
    % =====================================================
    prefix = sprintf('Data_p_%d_n_%d_design_%s_rep_%d', ...
        p, n, design_type, rep_num);

    X = data.X;
    Y = data.Y;
    YX = [Y, X];   % n x (p+1), first column Y

    writematrix(YX, fullfile(datadir, sprintf('%s.csv', prefix)));

    % =====================================================
    % Save true quantities only once
    % =====================================================
    save_beta_once_counter = save_beta_once_counter + 1;

    if save_beta_once_counter == 1
        % -----------------------------
        % Save tau grid (K x 1 vector)
        % -----------------------------
        prefix = sprintf('Data_tau_grid_p_%d_n_%d_design_%s', ...
            p, n, design_type);
        
        fname = fullfile(datadir, sprintf('%s.csv', prefix));
        writematrix(tau_grid_for_TRUE_generation(:), fname);
        
        % -----------------------------
        % True support: p x K matrix
        % -----------------------------
        prefix = sprintf('Data_TRUE_support_p_%d_n_%d_design_%s', ...
            p, n, design_type);

        true_support = data.true_support;
        fname = fullfile(datadir, sprintf('%s.csv', prefix));
        writematrix(double(true_support), fname);

        % -----------------------------
        % True beta: p x K matrix
        % -----------------------------
        prefix = sprintf('Data_TRUE_beta_p_%d_n_%d_design_%s', ...
            p, n, design_type);

        beta_true_grid = data.beta_true_grid;
        fname = fullfile(datadir, sprintf('%s.csv', prefix));
        writematrix(beta_true_grid, fname);

        % -----------------------------
        % True beta0: K x 1 vector
        % -----------------------------
        prefix = sprintf('Data_TRUE_beta0_p_%d_n_%d_design_%s', ...
            p, n, design_type);

        beta0_true_grid = data.beta0_true_grid;
        fname = fullfile(datadir, sprintf('%s.csv', prefix));
        writematrix(beta0_true_grid(:), fname);
    end
end