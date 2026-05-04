clear; clc;

%% =========================
% USER INPUT
%% =========================
n = 200;
p = 50;

design_type = 'corr';
rep_range = 1:20;
outdir = 'Output';

% Choose one of:
%   'UNIQUE'
%   'BONDELL'
%   'ALL'
method_used = 'ALL';

%% =========================
% METRICS
%% =========================
metric_cols = { ...
    'TPR','FPR','MCC','FDR','beta_RMSE','Q_RMSE', ...
    'n_true','comp_time','n_same','n_opposite','n_zero'};

all_tables = {};
used_reps = [];

%% =========================
% LOAD FILES
%% =========================
if strcmpi(method_used, 'UNIQUE')

    % -----------------------------------------------------
    % Load UNIQUE files only
    % -----------------------------------------------------
    for rep = rep_range

        fname = fullfile(outdir, sprintf( ...
            'Output_summary_p_%d_n_%d_design_%s_rep_%d.csv', ...
            p, n, design_type, rep));

        if exist(fname, 'file')
            T = readtable(fname, 'VariableNamingRule','preserve');

            if ~ismember('Row', T.Properties.VariableNames)
                error('File %s does not contain a "Row" column.', fname);
            end

            all_tables{end+1} = T; %#ok<SAGROW>
            used_reps(end+1) = rep; %#ok<SAGROW>
        end
    end

    if isempty(all_tables)
        error('No UNIQUE files found!');
    end

    fprintf('\nLoaded %d UNIQUE files. Reps used: %s\n', ...
        length(used_reps), mat2str(used_reps));

elseif strcmpi(method_used, 'BONDELL')

    % -----------------------------------------------------
    % Load BONDELL files only
    % -----------------------------------------------------
    for rep = rep_range

        fname = fullfile(outdir, sprintf( ...
            'Output_summary_BONDELL_p_%d_n_%d_design_%s_rep_%d.csv', ...
            p, n, design_type, rep));

        if exist(fname, 'file')
            T = readtable(fname, 'VariableNamingRule','preserve');

            if ~ismember('Row', T.Properties.VariableNames)
                error('File %s does not contain a "Row" column.', fname);
            end

            all_tables{end+1} = T; %#ok<SAGROW>
            used_reps(end+1) = rep; %#ok<SAGROW>
        end
    end

    if isempty(all_tables)
        error('No BONDELL files found!');
    end

    fprintf('\nLoaded %d BONDELL files. Reps used: %s\n', ...
        length(used_reps), mat2str(used_reps));

elseif strcmpi(method_used, 'ALL')

    % -----------------------------------------------------
    % Load UNIQUE files
    % -----------------------------------------------------
    tables_unique = {};
    reps_unique = [];

    for rep = rep_range

        fname_u = fullfile(outdir, sprintf( ...
            'Output_summary_p_%d_n_%d_design_%s_rep_%d.csv', ...
            p, n, design_type, rep));

        if exist(fname_u, 'file')
            T_u = readtable(fname_u, 'VariableNamingRule','preserve');

            if ~ismember('Row', T_u.Properties.VariableNames)
                error('UNIQUE file %s does not contain a "Row" column.', fname_u);
            end

            tables_unique{end+1} = T_u; %#ok<SAGROW>
            reps_unique(end+1) = rep; %#ok<SAGROW>
        end
    end

    % -----------------------------------------------------
    % Load BONDELL files
    % -----------------------------------------------------
    tables_bondell = {};
    reps_bondell = [];

    for rep = rep_range

        fname_b = fullfile(outdir, sprintf( ...
            'Output_summary_BONDELL_p_%d_n_%d_design_%s_rep_%d.csv', ...
            p, n, design_type, rep));

        if exist(fname_b, 'file')
            T_b = readtable(fname_b, 'VariableNamingRule','preserve');

            if ~ismember('Row', T_b.Properties.VariableNames)
                error('BONDELL file %s does not contain a "Row" column.', fname_b);
            end

            tables_bondell{end+1} = T_b; %#ok<SAGROW>
            reps_bondell(end+1) = rep; %#ok<SAGROW>
        end
    end

    % -----------------------------------------------------
    % Keep only common reps
    % -----------------------------------------------------
    common_reps = intersect(reps_unique, reps_bondell);

    if isempty(common_reps)
        error('No common rep files found across UNIQUE and BONDELL.');
    end

    fprintf('\nLoaded %d common rep files for ALL. Reps used: %s\n', ...
        length(common_reps), mat2str(common_reps));

    all_tables = cell(1, numel(common_reps));
    used_reps = common_reps;

    for ii = 1:numel(common_reps)
        rep = common_reps(ii);

        idx_u = find(reps_unique == rep, 1);
        idx_b = find(reps_bondell == rep, 1);

        if isempty(idx_u) || isempty(idx_b)
            error('Internal matching error for rep %d.', rep);
        end

        T_unique = tables_unique{idx_u};
        T_bondell = tables_bondell{idx_b};

        if ~ismember('Row', T_unique.Properties.VariableNames)
            error('UNIQUE file for rep %d does not contain a "Row" column.', rep);
        end

        if ~ismember('Row', T_bondell.Properties.VariableNames)
            error('BONDELL file for rep %d does not contain a "Row" column.', rep);
        end

        % Keep UNIQUE rows on top, BONDELL appended below
        T_combined = [T_unique; T_bondell];
        all_tables{ii} = T_combined;
    end

else
    error('method_used must be one of: ''UNIQUE'', ''BONDELL'', ''ALL''.');
end

%% =========================
% INITIALIZE
%% =========================
if isempty(all_tables)
    error('No files found!');
end

T0 = all_tables{1};

if ~ismember('Row', T0.Properties.VariableNames)
    error('First loaded file does not contain a "Row" column.');
end

methods = string(T0.Row);     % e.g. UNIQUE, QRLASSO, BONDELL
n_methods = numel(methods);
R = numel(all_tables);

data = struct();
for j = 1:numel(metric_cols)
    data.(metric_cols{j}) = nan(n_methods, R);
end

%% =========================
% STACK DATA
%% =========================
for r = 1:R

    T = all_tables{r};

    if ~ismember('Row', T.Properties.VariableNames)
        error('File for rep %d does not contain a "Row" column.', used_reps(r));
    end

    [tf, loc] = ismember(methods, string(T.Row));
    if ~all(tf)
        error('Method names in rep %d do not match the first file.', used_reps(r));
    end
    T = T(loc, :);

    for j = 1:numel(metric_cols)
        col = metric_cols{j};

        if ~ismember(col, T.Properties.VariableNames)
            error('Column %s missing in rep %d.', col, used_reps(r));
        end

        vals = T.(col);
        if iscell(vals) || isstring(vals)
            vals = str2double(string(vals));
        end

        data.(col)(:, r) = vals(:);
    end
end

%% =========================
% RECALCULATE SIGN RATIOS AMONG SELECTED NONZERO SIGN-CHECKED ENTRIES
%% =========================
denom_selected = data.n_same + data.n_opposite;

data.ratio_same_in_selected = data.n_same ./ denom_selected;
data.ratio_opposite_in_selected = data.n_opposite ./ denom_selected;

data.ratio_same_in_selected(denom_selected == 0) = NaN;
data.ratio_opposite_in_selected(denom_selected == 0) = NaN;

metric_cols = { ...
    'TPR','FPR','MCC','FDR','beta_RMSE','Q_RMSE', ...
    'ratio_same_in_selected','ratio_opposite_in_selected', ...
    'n_true','comp_time','n_same','n_opposite','n_zero'};

%% =========================
% BUILD FINAL TABLE
% rows = methods
% cols = metrics
%% =========================
T_out = table(cellstr(methods), 'VariableNames', {'Method'});

for j = 1:numel(metric_cols)

    col = metric_cols{j};
    X = data.(col);   % n_methods x R

    mu = mean(X, 2, 'omitnan');
    se = std(X, 0, 2, 'omitnan') ./ sqrt(sum(~isnan(X), 2));

    out_str = strings(n_methods, 1);

    for i = 1:n_methods

        if ismember(col, {'TPR','FPR','MCC','FDR','beta_RMSE','Q_RMSE', ...
        'ratio_same_in_selected','ratio_opposite_in_selected'})
            out_str(i) = sprintf('%.2f(%.3f)', mu(i), se(i));

        elseif ismember(col, {'comp_time','n_true','n_same','n_opposite','n_zero'})
            out_str(i) = sprintf('%.1f(%.2f)', mu(i), se(i));

        else
            out_str(i) = sprintf('%.3f(%.4f)', mu(i), se(i));
        end
    end

    T_out.(col) = cellstr(out_str);
end

%% =========================
% DISPLAY
%% =========================
disp(T_out);

%% =========================
% SAVE OUTPUT
%% =========================
if strcmpi(method_used, 'BONDELL')

    outname = fullfile(outdir, sprintf( ...
        'Output_FULL_summary_MEAN_SE_BONDELL_p_%d_n_%d_design_%s.csv', ...
        p, n, design_type));

elseif strcmpi(method_used, 'ALL')

    outname = fullfile(outdir, sprintf( ...
        'Output_FULL_summary_MEAN_SE_ALL_p_%d_n_%d_design_%s.csv', ...
        p, n, design_type));

else

    outname = fullfile(outdir, sprintf( ...
        'Output_FULL_summary_MEAN_SE_UNIQUE_p_%d_n_%d_design_%s.csv', ...
        p, n, design_type));
end

writetable(T_out, outname);

fprintf('\nSaved summary file:\n%s\n', outname);