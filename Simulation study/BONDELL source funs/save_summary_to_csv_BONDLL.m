function save_summary_to_csv_BONDLL(summary, p, n, design_type, rep_num, outdir)

    if nargin < 6 || isempty(outdir)
        outdir = 'Output';
    end

    if ~exist(outdir, 'dir')
        mkdir(outdir);
    end

    % ---------------------------------------------------------
    % Extract metric names from BONDELL
    % ---------------------------------------------------------
    metric_names = fieldnames(summary.BONDELL);
    n_metrics = numel(metric_names);

    % ---------------------------------------------------------
    % Fill matrix
    % ---------------------------------------------------------
    M = zeros(1, n_metrics);
    row_names = {'BONDELL'};

    for j = 1:n_metrics
        M(1,j) = summary.BONDELL.(metric_names{j});
    end

    % ---------------------------------------------------------
    % Convert to table
    % ---------------------------------------------------------
    T = array2table(M, 'VariableNames', metric_names);
    T.Properties.RowNames = row_names;

    % ---------------------------------------------------------
    % File name
    % ---------------------------------------------------------
    fname = sprintf('Output_summary_BONDELL_p_%d_n_%d_design_%s_rep_%d.csv', ...
        p, n, design_type, rep_num);

    fullpath = fullfile(outdir, fname);

    % ---------------------------------------------------------
    % Write CSV with row names
    % ---------------------------------------------------------
    writetable(T, fullpath, 'WriteRowNames', true);

    fprintf('Summary saved: %s\n', fullpath);
end