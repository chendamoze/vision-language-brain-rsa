import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pingouin import partial_corr
from scipy.stats import pearsonr, wilcoxon
from statsmodels.stats.multitest import multipletests


def compute_partial(brain_vec, dnn1, dnn2, dnn3):
    """
    Compute the partial correlation between a brain-vector and a DNN representation,
    while controlling for two other DNN representations.
    """
    df = pd.DataFrame({
        'brain': brain_vec,
        'dnn1': dnn1,
        'dnn2': dnn2,
        'dnn3': dnn3
    })
    return partial_corr(data=df, x='brain', y='dnn1', covar=['dnn2', 'dnn3'], method='pearson')

def fmri_to_group(roi):
    """
    Map an ROI name to its anatomical group (Early, Dorsal, Ventral).
    """
    if roi in ['V1d', 'V1v', 'V2d', 'V2v', 'V3d', 'V3v']:
        return 'Early'
    elif roi in ['IPS0', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'SPL1', 'FEF', 'V3a', 'V3b', 'MST', 'hMT']:
        return 'Dorsal'
    elif roi in ['LO1', 'LO2', 'VO1', 'VO2', 'PHC1', 'PHC2', 'hV4']:
        return 'Ventral'
    else:
        return None

def get_significance_stars(p_value):
    """
    Convert a p-value into common significance star notation.
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'n.s'

def create_upper_triangle_comb(filepath, comb, brain_vectors, excluded):
    """
    Builds RDM vectors by extracting the upper triangle of ROI × subject matrices.
    Excludes images flagged as invalid.
    """
    df = pd.read_csv(filepath)
    grouped = df.groupby(comb)
    matrices = {}
    excluded = {int(x) for x in excluded}
    for keys, group in grouped:
        images = sorted(set(group["image1"]).union(set(group["image2"])) - excluded)
        image_to_idx = {img: i for i, img in enumerate(images)}
        n = len(images)
        mat = np.full((n, n), np.nan)
        for _, row in group.iterrows():
            if row["image1"] in excluded or row["image2"] in excluded:
                continue
            i = image_to_idx[row["image2"]]
            j = image_to_idx[row["image1"]]
            mat[i, j] = row[brain_vectors]
        upper_triangle = mat[np.triu_indices(n, k=1)]
        matrices[keys] = upper_triangle
    return matrices

def csv_to_dnn_vector(filepath):
    """
    Load a DNN model vector from CSV where each row contains a 'score' column.
    """
    df = pd.read_csv(filepath)
    return df['score'].to_numpy()


def add_significance_annotations(ax, df_avg, sig_df, group, corr_type, representations_labels, means):
    """
    Add both one-sample and paired significance markers above bar plots.
    """

    # One-sample (vs 0)
    for i, rep in enumerate(representations_labels):
        p_row = sig_df[
            (sig_df['group'] == group) &
            (sig_df['corr_type'] == corr_type) &
            (sig_df['test_type'] == 'one_sample') &
            (sig_df['comparison'] == f'{rep} vs 0')
        ]
        if not p_row.empty:
            signif = get_significance_stars(p_row['p_corrected'].values[0])
            ax.text(i, means[i] + 0.01, signif, ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Paired comparisons 
    paired_comparisons = [
        (0,1), (0,2), (1,2)
    ]
    base_offset = 0.03
    extra_offset = 0.015  
    used_heights = []

    for line_idx, (idx1, idx2) in enumerate(paired_comparisons):
        rep1, rep2 = representations_labels[idx1], representations_labels[idx2]
        p_row = sig_df[
            (sig_df['group'] == group) &
            (sig_df['corr_type'] == corr_type) &
            (sig_df['test_type'] == 'paired') &
            (sig_df['comparison'] == f'{rep1} vs {rep2}')
        ]
        if not p_row.empty:
            signif = get_significance_stars(p_row['p_corrected'].values[0])
            y = max(means[idx1], means[idx2]) + base_offset

            while any(abs(y - h) < 1e-6 for h in used_heights):
                y += extra_offset
            used_heights.append(y)

            ax.plot([idx1, idx2], [y, y], color='black', linewidth=1)
            ax.text((idx1 + idx2)/2, y + 0.005, signif, ha='center', va='bottom', fontsize=14, fontweight='bold')


def fmri_plot(fmri_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec):
    """
    Compute Zero-Order and Partial correlations between fMRI RDMs and three
    types of DNN-based RDMs, run statistical tests, generate bar plots with
    significance annotations, and save results.
    """

    groups = ['Early', 'Dorsal', 'Ventral']
    representations_labels = ['Images', 'Visual Text', 'Abstract Text']
    data_records = []

    for (subj, roi), brain_vec in fmri_matrices.items():
        group = fmri_to_group(roi)
        if group is None:
            continue

        # partial correlations
        partials = {
            'Images': compute_partial(brain_vec, dnn_image_vec, dnn_visual_vec, dnn_semantic_vec)['r'].values[0],
            'Visual Text': compute_partial(brain_vec, dnn_visual_vec, dnn_image_vec, dnn_semantic_vec)['r'].values[0],
            'Abstract Text': compute_partial(brain_vec, dnn_semantic_vec, dnn_image_vec, dnn_visual_vec)['r'].values[0]
        }

        for rep_label, dnn_vec in zip(representations_labels,
                                      [dnn_image_vec, dnn_visual_vec, dnn_semantic_vec]):
            r, _ = pearsonr(brain_vec, dnn_vec)
            data_records.append({
                'subj': subj,
                'roi': roi,
                'Group': group,
                'Representation': rep_label,
                'Zero': r,
                'Partial': partials[rep_label]
            })

    df_all = pd.DataFrame(data_records)

    df_avg = df_all.groupby(['subj', 'Group', 'Representation'], as_index=False).mean(numeric_only=True)

    # --- Statistical tests---
    all_pvalues = []
    test_info = []
    stats_results = {}

    for group in groups:
        df_group = df_avg[df_avg['Group'] == group]
        stats_results[group] = {'Zero': {}, 'Partial': {}}

        for corr_type in ['Zero', 'Partial']:
            one_sample_results = {}
            paired_results = {}

            # One-sample Wilcoxon 
            for rep in representations_labels:
                values = df_group[df_group['Representation'] == rep][corr_type].values
                if len(values) > 0:
                    stat, p = wilcoxon(values, alternative='greater')
                    one_sample_results[rep] = p
                    all_pvalues.append(p)
                    test_info.append({
                        'group': group,
                        'corr_type': corr_type,
                        'test_type': 'one_sample',
                        'comparison': f'{rep} vs 0',
                        'p_value': p
                    })

            # Paired Wilcoxon 
            comparison_pairs = [
                ('Images', 'Visual Text'),
                ('Images', 'Abstract Text'),
                ('Visual Text', 'Abstract Text')
            ]
            for rep1, rep2 in comparison_pairs:
                subj_shared = set(df_group[df_group['Representation'] == rep1]['subj']) & \
                              set(df_group[df_group['Representation'] == rep2]['subj'])
                vals1 = df_group[(df_group['Representation'] == rep1) &
                                 (df_group['subj'].isin(subj_shared))][corr_type].values
                vals2 = df_group[(df_group['Representation'] == rep2) &
                                 (df_group['subj'].isin(subj_shared))][corr_type].values
                if len(vals1) > 0 and len(vals1) == len(vals2):
                    stat, p = wilcoxon(vals1, vals2)
                    paired_results[(rep1, rep2)] = p
                    all_pvalues.append(p)
                    test_info.append({
                        'group': group,
                        'corr_type': corr_type,
                        'test_type': 'paired',
                        'comparison': f'{rep1} vs {rep2}',
                        'p_value': p
                    })

            stats_results[group][corr_type] = {
                'one_sample': one_sample_results,
                'paired': paired_results
            }

    # FDR correction
    reject, pvals_corrected, _, _ = multipletests(all_pvalues, method='fdr_bh')
    for i, info in enumerate(test_info):
        info['p_corrected'] = pvals_corrected[i]
        info['significant'] = reject[i]

    stats_df = pd.DataFrame(test_info)
    os.makedirs("fmri_correlation_scores", exist_ok=True)
    stats_df.to_csv("fmri_correlation_scores/statistical_tests_results.csv", index=False)

    
    plots_dir = "fmri_plots"
    os.makedirs(plots_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
    custom_palette = {"Images": "#2c2c8d", "Visual Text": "#8e44ad", "Abstract Text": "#b33939"}

    sig_df = pd.DataFrame(test_info)

    for row_idx, corr_type in enumerate(['Zero', 'Partial']):
        for col_idx, group in enumerate(groups):
            ax = axes[row_idx, col_idx]
            df_group = df_avg[df_avg['Group'] == group]
            means, sems = [], []

            for rep in representations_labels:
                vals = df_group[df_group['Representation'] == rep][corr_type].values
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))

            x = np.arange(len(representations_labels))
            colors = [custom_palette[r] for r in representations_labels]
            bars = ax.bar(x, means, yerr=sems, color=colors, capsize=5, edgecolor='black')

            # Mark values above the columns
            for bar, mean in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, mean/2, f"{mean:.2f}",
                        ha='center', va='center', color='white', fontsize=12, fontweight='bold')

            # Add significance
            add_significance_annotations(ax, df_group, sig_df, group, corr_type, representations_labels, means)

            ax.set_xticks(x)
            ax.set_xticklabels(representations_labels)
            ax.set_ylim(0, 0.25)
            if col_idx == 0:
                ax.set_ylabel(f"{corr_type} Correlation", fontsize=12, fontweight='bold')
            if row_idx == 0:
                ax.set_title(group, fontsize=14, fontweight='bold')

    plt.suptitle("fMRI Correlation Analysis (Participant-level, FDR-corrected)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "combined_correlation_analysis.png"), dpi=300)
    plt.close()

    print("Analysis complete!")
    print("Results saved to:")
    print("fmri_correlation_scores/statistical_tests_results.csv")
    print("fmri_plots/combined_correlation_analysis.png")

