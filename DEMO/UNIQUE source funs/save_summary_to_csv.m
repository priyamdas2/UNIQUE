function save_summary_to_csv(summary, p, n, design_type, rep_num, outdir)

    if nargin < 6 || isempty(outdir)
        outdir = 'Output';
    end

    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    % ---------------------------------------------------------
    % Extract metric names from UNIQUE (assume same for QRLASSO)
    % ---------------------------------------------------------
    metric_names = fieldnames(summary.UNIQUE);
    n_metrics = numel(metric_names);

    % ---------------------------------------------------------
    % Initialize matrix
    % ---------------------------------------------------------
    has_qrlasso = isfield(summary, 'QRLASSO');

    if has_qrlasso
        M = zeros(2, n_metrics);
        row_names = {'UNIQUE', 'QRLASSO'};
    else
        M = zeros(1, n_metrics);
        row_names = {'UNIQUE'};
    end

    % ---------------------------------------------------------
    % Fill UNIQUE row
    % ---------------------------------------------------------
    for j = 1:n_metrics
        M(1,j) = summary.UNIQUE.(metric_names{j});
    end

    % ---------------------------------------------------------
    % Fill QRLASSO row (if exists)
    % ---------------------------------------------------------
    if has_qrlasso
        for j = 1:n_metrics
            if isfield(summary.QRLASSO, metric_names{j})
                M(2,j) = summary.QRLASSO.(metric_names{j});
            else
                M(2,j) = NaN;  % in case some fields missing
            end
        end
    end

    % ---------------------------------------------------------
    % Convert to table
    % ---------------------------------------------------------
    T = array2table(M, 'VariableNames', metric_names);
    T.Properties.RowNames = row_names;

    % ---------------------------------------------------------
    % File name
    % ---------------------------------------------------------
    fname = sprintf('Output_summary_p_%d_n_%d_design_%s_rep_%d.csv', ...
        p, n, design_type, rep_num);

    fullpath = fullfile(outdir, fname);

    % ---------------------------------------------------------
    % Write CSV with row names
    % ---------------------------------------------------------
    writetable(T, fullpath, 'WriteRowNames', true);

    fprintf('Summary saved: %s\n', fullpath);
end