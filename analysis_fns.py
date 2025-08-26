import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(context='paper', style='whitegrid', font='serif', font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    # Fallback to generic serif if Times New Roman is not available
    "font.serif": ["Times New Roman", "serif"],
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def read_csv_with_tag(file_path, tag):
    
    df = pd.read_csv(file_path, header=0)    
    df['tag'] = tag

    df = df.loc[:, ~df.columns.str.contains('MIN|MAX')]
    df.columns = [f"ratio-{col.split('format-')[1].split('- val')[0].strip()}" if 'format-' in col else col for col in df.columns]

    return df

def read_csv_with_tag_vanilla(file_path, tag):
    df = pd.read_csv(file_path, header=0)
    df['tag'] = tag

    # remove columns that contain 'MIN' or 'MAX' in their names
    df = df.loc[:, ~df.columns.str.contains('MIN|MAX')]
    
    # rename columns to remove 'format-' prefix and '- val' suffix
    df.columns = [f"ratio-{col.split('format-')[1].split('-vanilla')[0].strip()}" if 'format-' in col else col for col in df.columns]

    return df

def read_csv_with_tag_optimization(file_path, tag):
    df = pd.read_csv(file_path, header=0)
    df['tag'] = tag

    # remove columns that contain 'MIN' or 'MAX' in their names
    df = df.loc[:, ~df.columns.str.contains('MIN|MAX')]

    df.columns = [f"ratio-{col.split('format-')[1].split('- val')[0].strip()}" if 'format-' in col else col for col in df.columns]

    return df

def extract_data(df, column, tags):
    df_new = df[[column, 'tag']][df['tag'].isin(tags)]
    
    df_final = pd.DataFrame()
    for tag in tags:
        df_tag = df_new[df_new['tag'] == tag].drop(columns='tag').reset_index(drop=True)
        df_tag.columns = [f"{column}-{tag}"]
        df_final = pd.concat([df_final, df_tag], axis=1)
    
    # remove rows where all values are NaN
    df_final = df_final.dropna(how='all').reset_index(drop=True)

    return df_final

def pareto_front(df: pd.DataFrame, maximize: tuple | list | None = None) -> pd.Index:
    if maximize is None:
        maximize = [False] * df.shape[1]
    if len(maximize) != df.shape[1]:
        raise ValueError("`maximize` must match the number of objectives")

    data = df.to_numpy(copy=False)
    signs = np.where(maximize, -1.0, 1.0)            # maximise -> negate
    data = data * signs

    # Handle NaNs (treat them as inf so the row is dominated)
    data = np.nan_to_num(data, nan=np.inf)

    # Pairwise domination test (broadcasted)
    # row i dominates row j  <->  all(data[i] ≤ data[j]) and at least one <
    less_equal = data[:, None] <= data[None, :]
    strictly_less = data[:, None] < data[None, :]
    dominates = less_equal.all(axis=2) & strictly_less.any(axis=2)

    # A row is non-dominated if no other row dominates it
    is_dominated = dominates.any(axis=0)
    pareto_mask = ~is_dominated

    return df.index[pareto_mask]


def draw_pareto_front(df, front, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.scatter(df[xlabel], df[ylabel], c='lightgray', label='All points', alpha=0.5)
    plt.scatter(df.loc[front, xlabel], df.loc[front, ylabel], c='red', label='Pareto front', edgecolor='black')
    
    min_index = front.min()
    # label the minimum point on the Pareto front with number of its index
    plt.scatter(df.loc[min_index, xlabel], df.loc[min_index, ylabel], c='yellow', label='Minimum Steps', edgecolor='black', s=100)
    plt.annotate(f'{min_index}', 
                 (df.loc[min_index, xlabel], df.loc[min_index, ylabel]), 
                 textcoords="offset points",
                 xytext=(15,-1),
                 ha='center', fontsize=9, color='black')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Pareto Front')
    plt.legend()
    plt.grid()
    plt.show()

def draw_multiple_pareto_fronts(df, columns, tags, maximize, xlim=None, ylim=None):

    plt.figure(figsize=(10, 6))
    
    for col in columns:
        df_new = extract_data(df, col, tags)

        for tag in df_new.columns:
            if 'conciseness' in tag:
                df_new[tag] = np.log10(df_new[tag])

        init_point = df_new.iloc[0, :].values
    
        curr_front = pareto_front(df_new, maximize=maximize)

        plt.scatter(df_new[f'{col}-{tags[0]}'], df_new[f'{col}-{tags[1]}'], c='lightgray', alpha=0.5)
        
        
        # reorder the data so that the points are connected in the order of the Pareto front
        ordered_df = df_new.loc[curr_front].sort_values(by=f'{col}-{tags[0]}')
        plt.plot(ordered_df[f'{col}-{tags[0]}'], ordered_df[f'{col}-{tags[1]}'], label=f'Pareto Front - {col}', linewidth=2, marker='o')

        min_index = curr_front.min()
        avg_index = round(np.mean(curr_front))
        plt.scatter(df_new.loc[min_index, f'{col}-{tags[0]}'], df_new.loc[min_index, f'{col}-{tags[1]}'], c='yellow', edgecolor='black', s=100)
        plt.annotate(f'{avg_index}',
                        (df_new.loc[min_index, f'{col}-{tags[0]}'], df_new.loc[min_index, f'{col}-{tags[1]}']), 
                        textcoords="offset points",
                        xytext=(15,-1),
                        ha='center', fontsize=9, color='black')
    
    plt.scatter(init_point[0], init_point[1], c='blue', label='Init Model', edgecolor='black', s=100)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.xlabel(tags[0])
    plt.ylabel(tags[1])
    plt.title('Pareto Fronts')
    plt.legend()
    plt.grid()
    plt.show()

def compute_pareto_front_stats(df, columns, tags, maximize):

    result = {}
    for col in columns:
        df_new = extract_data(df, col, tags)

        curr_front = pareto_front(df_new, maximize=maximize)
        df_pareto = df_new.loc[curr_front]
        
        min_stats = df_pareto.min()
        max_stats = df_pareto.max()
        mean_stats = df_pareto.mean()

        result[col] = {
            "min": min_stats,
            "max": max_stats,
            "mean": mean_stats
        }
    
    result_efficiency = {}
    result_accuracy = {}
    result_balanced = {}

    for key, value in result.items():
        if "Efficiency" in key:
            result_efficiency[key] = value
        elif "Accuracy" in key:
            result_accuracy[key] = value
        elif "Balanced" in key:
            result_balanced[key] = value

    for dict in [result_efficiency, result_accuracy, result_balanced]:
        if len(dict) == 2:
            keys = list(dict.keys())
            assert "ours" in keys[0] and "baseline" in keys[1]
            ours_dict = dict[keys[0]]
            baseline_dict = dict[keys[1]]

            assert ours_dict.keys() == baseline_dict.keys()

            for key in ours_dict.keys():
                print(f"--------------------------------{key}--------------------------------")
                print(f"ours: {ours_dict[key]}")
                print(f"baseline: {baseline_dict[key]}")
        elif len(dict) == 1:
            for key, value in dict.items():
                for k, v in value.items():
                    print(f"--------------------------------{k}--------------------------------")
                    print(f"{v}")
        else:
            raise ValueError("Invalid number of keys in the dictionary")

def draw_pareto_parallel_coordinates(df, columns, tags, maximize, save_path=None):

    df_combined = pd.DataFrame()
    for col in columns:
        df_new = extract_data(df, col, tags)

        df_combined = pd.concat([df_combined, df_new], axis=1)

    df_combined = df_combined.dropna(how="all")
    if df_combined.empty:
        raise ValueError("No data available for the requested columns/tags.")

    assert len(tags) == len(maximize), "Number of tags and maximize must match"

    # normalise the data to the range [0, 1]
    norm_min = df_combined.min()
    norm_range = df_combined.max() - df_combined.min()
    norm_range.replace(0, 1, inplace=True)  # avoid division by zero
    normalized_all = (df_combined - norm_min) / norm_range

    plt.figure(figsize=(5.5, 3))

    x_positions = np.arange(len(tags))
    display_tags = [t.capitalize() for t in tags]
    display_tags = [t.replace("Format", "Clarity") for t in display_tags]
    
    plt.xticks(x_positions, display_tags, rotation=0, fontsize=14)
    plt.yticks(fontsize=14)

    avg_index_dict = {}

    non_baseline_cols = [c for c in columns if "baseline" not in c.lower()]
    if len(non_baseline_cols) <= 10:
        palette_cmap = plt.cm.get_cmap("tab10", len(non_baseline_cols))
    elif len(non_baseline_cols) <= 20:
        palette_cmap = plt.cm.get_cmap("tab20", len(non_baseline_cols))
    else:
        palette_cmap = plt.cm.get_cmap("hsv", len(non_baseline_cols))

    palette_colors = {col: palette_cmap(i) for i, col in enumerate(non_baseline_cols)}

    for col_idx, col in enumerate(columns):
        group_cols = [f"{col}-{t}" for t in tags]

        front_idx_col = pareto_front(df_combined[group_cols], maximize=maximize)

        group_data = normalized_all.loc[front_idx_col, group_cols]

        if "baseline" in col.lower():
            color = "#6c757d"  # muted grey for baselines
        else:
            color = palette_colors.get(col, "#1f77b4")

        is_baseline = "baseline" in col.lower()
        line_alpha = 0.7 if is_baseline else 1.0
        line_width = 1.5 if is_baseline else 1.5
        style = "s--" if is_baseline else "o-"
        marker_size = 5 if is_baseline else 7

        x_offset_line = -0.03 if is_baseline else 0.03

        for _, row in group_data.iterrows():
            plt.plot(x_positions + x_offset_line, row.values, style, color=color, alpha=line_alpha,
                     linewidth=line_width, markersize=marker_size, label=col, zorder=3)
                     
        avg_index_dict[col] = round(np.mean(front_idx_col), 1)

        raw_stats = {}
        for t in group_cols:
            raw_stats[t] = df_combined.loc[front_idx_col, t].mean()

        x_shift_common = 0.12
        y_shift = 0.01 if not is_baseline else -0.01
        v_align = "bottom" if not is_baseline else "top"

        middle_idx = len(tags) // 2

        for idx, (tag_col, x_pos) in enumerate(zip(group_cols, x_positions)):
            best_idx = group_data[tag_col].idxmax()
            y_pos_norm = group_data.loc[best_idx, tag_col]
            raw_val = raw_stats[tag_col]

            if idx == middle_idx:
                y_shift_mid = 0.3 if not is_baseline else 0.27
                plt.text(x_pos, y_pos_norm + y_shift_mid, f"{raw_val:.2f}",
                         color=color, fontsize=14, ha="center", va=v_align, fontweight="bold")
            else:
                plt.text(x_pos + x_shift_common, y_pos_norm + y_shift, f"{raw_val:.2f}",
                         color=color, fontsize=14, ha="left", va=v_align, fontweight="bold")

    plt.ylabel("Normalized Objective Value", fontsize=14)
    plt.grid(True, alpha=0.3)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=11, loc="best", framealpha=0.2)

    plt.tight_layout(pad=0.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=3000)
    else:
        plt.show()

    return avg_index_dict

def draw_pareto_fronts_teaser_figure(df, columns, tags, maximize, save_path=None, add_legend=True):

    sns.set_style("whitegrid")

    plt.figure(figsize=(5, 5))

    non_baseline_cols = [c for c in columns if "baseline" not in c.lower()]
    if len(non_baseline_cols) <= 10:
        cmap = plt.cm.get_cmap("tab10", len(non_baseline_cols))
    elif len(non_baseline_cols) <= 20:
        cmap = plt.cm.get_cmap("tab20", len(non_baseline_cols))
    else:
        cmap = plt.cm.get_cmap("hsv", len(non_baseline_cols))
    colour_map = {col: cmap(i) for i, col in enumerate(non_baseline_cols)}

    # Separate marker shape assignment for baseline vs others
    def _base_name(col_name: str) -> str:
        return col_name.split('(')[0].strip()

    shape_cycle = ['o', 's', '^', 'D', 'P', 'v', 'X', '<', '>']

    # Shapes for our (non‐baseline) curves keyed by base name
    ours_shape_map: dict[str, str] = {}
    for col in non_baseline_cols:
        bname = _base_name(col)
        if bname not in ours_shape_map:
            ours_shape_map[bname] = shape_cycle[len(ours_shape_map) % len(shape_cycle)]

    # Shapes for baseline curves keyed by full column name to ensure uniqueness
    baseline_cols = [c for c in columns if "baseline" in c.lower()]
    baseline_shapes_cycle = ['s', '^', 'D', 'P', 'v', 'X', '<', '>']
    baseline_shape_map = {col: baseline_shapes_cycle[i % len(baseline_shapes_cycle)]
                          for i, col in enumerate(baseline_cols)}

    for col_idx, col in enumerate(columns):
        df_new = extract_data(df, col, tags)

        for tag in df_new.columns:
            if 'conciseness' in tag:
                df_new[tag] = np.log10(df_new[tag])

        init_point = df_new.iloc[0, :].values
    
        curr_front = pareto_front(df_new, maximize=maximize)

        grey_label = 'Suboptimal Checkpoints' if col_idx == 0 else None
        plt.scatter(df_new[f'{col}-{tags[0]}'],
                    df_new[f'{col}-{tags[1]}'],
                    c='lightgray',
                    alpha=0.5,
                    label=grey_label)
        
        # reorder the data so that the points are connected in the order of the Pareto front
        ordered_df = df_new.loc[curr_front].sort_values(by=f'{col}-{tags[0]}')

        is_baseline = "baseline" in col.lower()
        line_colour = "#6c757d" if is_baseline else colour_map.get(col, "#1f77b4")
        line_style = "-" if is_baseline else "-"
        marker_style = baseline_shape_map[col] if is_baseline else ours_shape_map[_base_name(col)]
        plt.plot(ordered_df[f'{col}-{tags[0]}'],
                 ordered_df[f'{col}-{tags[1]}'],
                 label=col,
                 linewidth=2,
                 marker=marker_style,
                 markersize=5,
                 color=line_colour,
                 linestyle=line_style)
    
    plt.scatter(init_point[0], init_point[1],
                c='yellow', marker='*', s=150,
                edgecolor='black', linewidth=0.8,
                label='Initial Model')

    xlabel_display = tags[0].capitalize()
    if xlabel_display == "Format":
        xlabel_display = "Clarity"
    elif xlabel_display == "Conciseness":
        xlabel_display = "Response Length(log10)"
    plt.xlabel(xlabel_display, fontsize=16)
    ylabel_display = tags[1].capitalize()
    if ylabel_display == "Format":
        ylabel_display = "Clarity"
    elif ylabel_display == "Conciseness":
        ylabel_display = "Response Length(log10)"
    plt.ylabel(ylabel_display, fontsize=16)

    plt.tick_params(axis='x', which='major', labelsize=13)
    plt.tick_params(axis='y', which='major', labelsize=13)
    
    if add_legend:
        plt.legend(fontsize=11, loc="best", framealpha=0.8)
    plt.grid(alpha=0.3)

    # Save or display the figure
    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=3000)
    else:
        plt.show()


def draw_pareto_fronts_teaser_figure_enlarged(df, columns, tags, maximize, xlim=None, ylim=None, save_path=None):

    sns.set_style("whitegrid")

    plt.figure(figsize=(5, 5))

    non_baseline_cols = [c for c in columns if "baseline" not in c.lower()]
    if len(non_baseline_cols) <= 10:
        cmap = plt.cm.get_cmap("tab10", len(non_baseline_cols))
    elif len(non_baseline_cols) <= 20:
        cmap = plt.cm.get_cmap("tab20", len(non_baseline_cols))
    else:
        cmap = plt.cm.get_cmap("hsv", len(non_baseline_cols))
    colour_map = {col: cmap(i) for i, col in enumerate(non_baseline_cols)}

    def _base_name(col_name: str) -> str:
        return col_name.split('(')[0].strip()

    shape_cycle = ['o', 's', '^', 'D', 'P', 'v', 'X', '<', '>']

    ours_shape_map: dict[str, str] = {}
    for col in non_baseline_cols:
        bname = _base_name(col)
        if bname not in ours_shape_map:
            ours_shape_map[bname] = shape_cycle[len(ours_shape_map) % len(shape_cycle)]

    baseline_cols = [c for c in columns if "baseline" in c.lower()]
    baseline_shapes_cycle = ['s', '^', 'D', 'P', 'v', 'X', '<', '>']
    baseline_shape_map = {col: baseline_shapes_cycle[i % len(baseline_shapes_cycle)]
                          for i, col in enumerate(baseline_cols)}

    for col_idx, col in enumerate(columns):
        df_new = extract_data(df, col, tags)

        for tag in df_new.columns:
            if 'conciseness' in tag:
                df_new[tag] = np.log10(df_new[tag])

        curr_front = pareto_front(df_new, maximize=maximize)

        grey_label = 'Suboptimal Checkpoints' if col_idx == 0 else None
        plt.scatter(df_new[f'{col}-{tags[0]}'],
                    df_new[f'{col}-{tags[1]}'],
                    c='lightgray', alpha=0.5, label=grey_label)

        ordered_df = df_new.loc[curr_front].sort_values(by=f'{col}-{tags[0]}')

        is_baseline = "baseline" in col.lower()
        line_colour = "#6c757d" if is_baseline else colour_map.get(col, "#1f77b4")
        line_style = "-" if is_baseline else "-"
        marker_style = baseline_shape_map[col] if is_baseline else ours_shape_map[_base_name(col)]

        # Use smaller marker size for baselines to reduce overlap
        marker_sz = 14 if is_baseline else 20  # baseline smaller

        plt.plot(ordered_df[f'{col}-{tags[0]}'],
                 ordered_df[f'{col}-{tags[1]}'],
                 label=col, linewidth=2, marker=marker_style, markersize=marker_sz,
                 color=line_colour, linestyle=line_style)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('#66b3ff')
        spine.set_linewidth(1.5)

    plt.grid(alpha=0.3)


    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=3000)
    else:
        plt.show()


def draw_pareto_fronts_figure_enlarged(df, columns, tags, maximize, xlim=None, ylim=None, save_path=None, add_legend=False):

    sns.set_style("whitegrid")

    plt.figure(figsize=(5, 3))

    non_baseline_cols = [c for c in columns if "baseline" not in c.lower()]
    if len(non_baseline_cols) <= 10:
        cmap = plt.cm.get_cmap("tab10", len(non_baseline_cols))
    elif len(non_baseline_cols) <= 20:
        cmap = plt.cm.get_cmap("tab20", len(non_baseline_cols))
    else:
        cmap = plt.cm.get_cmap("hsv", len(non_baseline_cols))
    colour_map = {col: cmap(i) for i, col in enumerate(non_baseline_cols)}

    # Separate marker shape assignment for baseline vs others
    def _base_name(col_name: str) -> str:
        return col_name.split('(')[0].strip()

    shape_cycle = ['o', 's', '^', 'D', 'P', 'v', 'X', '<', '>']

    ours_shape_map: dict[str, str] = {}
    for col in non_baseline_cols:
        bname = _base_name(col)
        if bname not in ours_shape_map:
            ours_shape_map[bname] = shape_cycle[len(ours_shape_map) % len(shape_cycle)]

    baseline_cols = [c for c in columns if "baseline" in c.lower()]
    baseline_shapes_cycle = ['s', '^', 'D', 'P', 'v', 'X', '<', '>']
    baseline_shape_map = {col: baseline_shapes_cycle[i % len(baseline_shapes_cycle)]
                          for i, col in enumerate(baseline_cols)}

    for col_idx, col in enumerate(columns):
        df_new = extract_data(df, col, tags)

        for tag in df_new.columns:
            if 'conciseness' in tag:
                df_new[tag] = np.log10(df_new[tag])

        curr_front = pareto_front(df_new, maximize=maximize)

        grey_label = 'Suboptimal Checkpoints' if col_idx == 0 else None
        plt.scatter(df_new[f'{col}-{tags[0]}'],
                    df_new[f'{col}-{tags[1]}'],
                    c='lightgray', alpha=0.5, label=grey_label)

        ordered_df = df_new.loc[curr_front].sort_values(by=f'{col}-{tags[0]}')

        is_baseline = "baseline" in col.lower()
        line_colour = "#6c757d" if is_baseline else colour_map.get(col, "#1f77b4")
        line_style = "-" if is_baseline else "-"
        marker_style = baseline_shape_map[col] if is_baseline else ours_shape_map[_base_name(col)]

        marker_sz = 8 if is_baseline else 8

        plt.plot(ordered_df[f'{col}-{tags[0]}'],
                 ordered_df[f'{col}-{tags[1]}'],
                 label=col, linewidth=2, marker=marker_style, markersize=marker_sz,
                 color=line_colour, linestyle=line_style)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('#66b3ff')
        spine.set_linewidth(1.5)

    plt.grid(alpha=0.3)

    xlabel_display = tags[0].capitalize()
    if xlabel_display == "Format":
        xlabel_display = "Clarity"
    elif xlabel_display == "Conciseness":
        xlabel_display = "Response Length(log10)"
    plt.xlabel(xlabel_display, fontsize=14)
    ylabel_display = tags[1].capitalize()
    if ylabel_display == "Format":
        ylabel_display = "Clarity"
    elif ylabel_display == "Conciseness":
        ylabel_display = "Response Length(log10)"
    plt.ylabel(ylabel_display, fontsize=14)

    plt.tick_params(axis='x', which='major', labelsize=13)
    plt.tick_params(axis='y', which='major', labelsize=13)

    if add_legend:
        plt.legend(fontsize=11, loc="best", framealpha=0.8)

    if save_path is not None:
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=3000)
    else:
        plt.show()