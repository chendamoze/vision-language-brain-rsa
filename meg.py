import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pingouin import partial_corr
from itertools import combinations
from scipy.stats import pearsonr, sem, wilcoxon 
from statsmodels.stats.multitest import fdrcorrection

# Color settings for plots
COLORS = ['#3366cc', '#800080', '#cc0000']
COLOR_MAP = {
    'Image': COLORS[0],
    'Visual Text': COLORS[1],
    'Abstract Text': COLORS[2]
}
REPRESENTATION_ORDER = ['Image', 'Visual Text', 'Abstract Text']
DNN_LABELS = ['Image', 'Visual Text', 'Abstract Text']


def compute_zero_order_r(brain_vec, dnn_vec):
    """
    Compute zero-order Pearson correlation between brain data and a DNN vector.
    """
    r, _ = pearsonr(brain_vec, dnn_vec)
    return r


def meg_plot(meg_matrices, dnn_image_vec, dnn_semantic_vec, dnn_visual_vec):
    """
    Perform MEG–DNN correlation analysis and statistical testing, and generate
    timecourse plots with FDR-corrected significance markers.
    """
    results_base = "meg_correlation_results_wilcoxon"
    plots_dir_meg = "meg_correlation_plots_wilcoxon"
    os.makedirs(results_base, exist_ok=True)
    os.makedirs(plots_dir_meg, exist_ok=True)
    
    print("Starting MEG analysis")
    
    # Compute zero-order correlations
    all_data = []
    for keys, brain_vec in meg_matrices.items():
        brain_vec = np.array(brain_vec)
        for label, dnn_vec in zip(DNN_LABELS, [dnn_image_vec, dnn_visual_vec, dnn_semantic_vec]):
            r_obs = compute_zero_order_r(brain_vec, dnn_vec)
            all_data.append({
                "subj": keys[0], "session": keys[1], "ms": keys[2],
                "Representation": label, "R": r_obs
            })

    df_all = pd.DataFrame(all_data)
    
    # Wilcoxon tests
    stats_results = []
    time_points = df_all['ms'].unique()
    rep_pairs = list(combinations(REPRESENTATION_ORDER, 2))

    for ms in time_points:
        df_ms = df_all[df_all['ms'] == ms]
        for rep1, rep2 in rep_pairs:
            data1_series = df_ms[df_ms['Representation'] == rep1].set_index('subj')['R']
            data2_series = df_ms[df_ms['Representation'] == rep2].set_index('subj')['R']
            
            data1, data2 = data1_series.align(data2_series, join='inner')
            
            if len(data1) < 5:
                continue

            stat, p_val = wilcoxon(data1, data2)
            stats_results.append({
                'ms': ms,
                'Comparison': (rep1, rep2),
                'p_raw': p_val
            })

    # FDR correction
    if not stats_results:
        print("Could not perform MEG statistical analysis")
        return

    df_stats = pd.DataFrame(stats_results)
    p_values_raw = df_stats['p_raw'].to_numpy()
    rejected, p_values_corrected = fdrcorrection(p_values_raw)
    
    df_stats['p_fdr_corrected'] = p_values_corrected
    df_stats['is_significant_fdr'] = rejected
    
    # שמירת התוצאות הסטטיסטיות
    output_path = os.path.join(results_base, "meg_statistical_comparisons_fdr.csv")
    df_stats.to_csv(output_path, index=False)
    print(f"MEG statistical comparison results saved to: {output_path}")

    # Create timecourse plot with significance markers
    df_summary = df_all.groupby(['ms', 'Representation']).agg(
        R_mean=('R', 'mean'),
        R_sem=('R', lambda x: sem(x, nan_policy='omit')),
    ).reindex(REPRESENTATION_ORDER, level=1).reset_index()

    fig, ax = plt.subplots(figsize=(12, 7))
    
    for label in REPRESENTATION_ORDER:
        data_label = df_summary[df_summary['Representation'] == label]
        ax.plot(data_label['ms'], data_label['R_mean'], label=label, color=COLOR_MAP[label])
        ax.fill_between(
            data_label['ms'],
            data_label['R_mean'] - data_label['R_sem'],
            data_label['R_mean'] + data_label['R_sem'],
            alpha=0.2, color=COLOR_MAP[label]
        )
    
    y_min, y_max = ax.get_ylim()
    sig_y_pos_base = y_min - 0.05 * (y_max - y_min)
    sig_y_step = 0.02 * (y_max - y_min)
    
    comparison_styles = {
        ('Image', 'Visual Text'): {'color': '#ff7f0e', 'y_offset': 0},      # Orange
        ('Image', 'Abstract Text'): {'color': '#2ca02c', 'y_offset': 1},     # Green
        ('Visual Text', 'Abstract Text'): {'color': '#d62728', 'y_offset': 2} # Red
    }

    df_significant = df_stats[df_stats['is_significant_fdr']].copy()
    df_significant['Comparison'] = df_significant['Comparison'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    for comp, style in comparison_styles.items():
        sig_times = df_significant[df_significant['Comparison'] == comp]['ms']
        if not sig_times.empty:
            y_pos = sig_y_pos_base + style['y_offset'] * sig_y_step
            ax.scatter(sig_times, [y_pos] * len(sig_times), color=style['color'], marker='s', s=10, 
                       label=f"p < 0.05 ({comp[0]} vs {comp[1]})")

    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--', label='Stimulus onset')
    ax.set_title("MEG - Zero-order correlations over time with Significance")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Correlation (r)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir_meg, "meg_wilcoxon_corr_sem.png"))
    plt.close()
    
    print("MEG plot with significance generated.")